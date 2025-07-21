//! Integration tests for http-bidir-comm

use std::sync::Arc;

use http_bidir_comm::ClientStats;
use http_bidir_comm::HttpServer;
use http_bidir_comm::TaskItem;
use http_bidir_comm::TaskResult;
use poem::handler;
use poem::http::StatusCode;
use poem::test::TestClient;
use poem::web::Data;
use poem::web::Json;
use poem::web::Path;
use poem::Endpoint;
use poem::EndpointExt;
use poem::IntoResponse;
use poem::Route;
use serde::Deserialize;
use serde::Serialize;
use similar_asserts::assert_eq;
use tracing::error;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct TestTask {
    command: String,
    data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct TestResult {
    output: String,
}

type AppState = HttpServer<TestTask, TestResult>;

#[handler]
async fn poll_task_handler(
    Path(client_id): Path<String>,
    Data(server): Data<&Arc<AppState>>,
) -> poem::Result<impl IntoResponse> {
    match server.poll_task_internal(&client_id).await {
        Ok(Some(task)) => Ok(Json(task).into_response()),
        Ok(None) => Ok(StatusCode::NO_CONTENT.into_response()),
        Err(e) => {
            error!("Error polling task: {}", e);
            Ok(StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
    }
}

#[handler]
async fn submit_result_handler(
    Data(server): Data<&Arc<AppState>>,
    Json(result): Json<TaskResult<TestResult>>,
) -> poem::Result<impl IntoResponse> {
    if let Err(e) = server.submit_result_internal(result).await {
        error!("Error submitting result: {}", e);
        return Ok(StatusCode::INTERNAL_SERVER_ERROR.into_response());
    }
    Ok(StatusCode::OK.into_response())
}

#[handler]
async fn get_stats_handler(
    Path(client_id): Path<String>,
    Data(server): Data<&Arc<AppState>>,
) -> poem::Result<Json<ClientStats>> {
    Ok(server
        .get_client_stats(&client_id)
        .await
        .map(Json)
        .ok_or(poem::error::NotFoundError)?)
}

fn create_routes(server: Arc<HttpServer<TestTask, TestResult>>) -> impl Endpoint {
    Route::new()
        .at("/poll/:client_id", poll_task_handler)
        .at("/submit", submit_result_handler)
        .at("/stats/:client_id", get_stats_handler)
        .data(server)
}

struct TestHarness<E: Endpoint> {
    server: Arc<HttpServer<TestTask, TestResult>>,
    client: TestClient<E>,
}

impl<E: Endpoint> TestHarness<E> {
    fn new(server: Arc<HttpServer<TestTask, TestResult>>, endpoint: E) -> Self {
        let client = TestClient::new(endpoint);
        Self { server, client }
    }
}

#[tokio::test]
async fn test_empty_queue_poll() {
    let server = Arc::new(HttpServer::new());
    let harness = TestHarness::new(server.clone(), create_routes(server));
    let resp = harness.client.get("/poll/client_empty").send().await;
    resp.assert_status(StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn test_task_submission_and_polling() {
    let server = Arc::new(HttpServer::new());
    let harness = TestHarness::new(server.clone(), create_routes(server));
    let task = TestTask {
        command: "test".to_string(),
        data: "data".to_string(),
    };
    let task_id = harness
        .server
        .enqueue_task_for_client("client1", task)
        .await
        .unwrap();

    let resp = harness.client.get("/poll/client1").send().await;
    resp.assert_status_is_ok();
    let polled_task: TaskItem<TestTask> = resp.json().await.value().deserialize();
    assert_eq!(polled_task.id, task_id);
}

#[tokio::test]
async fn test_successful_task_flow() {
    let server = Arc::new(HttpServer::new());
    let harness = TestHarness::new(server.clone(), create_routes(server));
    let task = TestTask {
        command: "test".to_string(),
        data: "hello".to_string(),
    };
    let task_id = harness
        .server
        .enqueue_task_for_client("client_ok", task.clone())
        .await
        .unwrap();

    let resp = harness.client.get("/poll/client_ok").send().await;
    resp.assert_status_is_ok();
    let task_item: TaskItem<TestTask> = resp.json().await.value().deserialize();
    assert_eq!(task_item.id, task_id);
    assert_eq!(task_item.data, task);

    let result = TaskResult::success(
        task_item.id,
        "client_ok".to_string(),
        TestResult {
            output: "world".to_string(),
        },
    );
    let resp = harness
        .client
        .post("/submit")
        .body_json(&result)
        .send()
        .await;
    resp.assert_status_is_ok();

    let stats_resp = harness.client.get("/stats/client_ok").send().await;
    stats_resp.assert_status_is_ok();
    let stats: ClientStats = stats_resp.json().await.value().deserialize();
    assert_eq!(stats.completed_tasks, 1);
    assert_eq!(stats.pending_tasks, 0);
}

#[tokio::test]
async fn test_task_failure_flow() {
    let server = Arc::new(HttpServer::new());
    let harness = TestHarness::new(server.clone(), create_routes(server));
    let task = TestTask {
        command: "fail".to_string(),
        data: "something".to_string(),
    };
    let task_id = harness
        .server
        .enqueue_task_for_client("client_fail", task)
        .await
        .unwrap();

    let resp = harness.client.get("/poll/client_fail").send().await;
    resp.assert_status_is_ok();
    let task_item: TaskItem<TestTask> = resp.json().await.value().deserialize();
    assert_eq!(task_item.id, task_id);

    let result: TaskResult<TestResult> =
        TaskResult::failure(task_item.id, "client_fail".to_string(), "It failed");
    let resp = harness
        .client
        .post("/submit")
        .body_json(&result)
        .send()
        .await;
    resp.assert_status_is_ok();

    let stats_resp = harness.client.get("/stats/client_fail").send().await;
    stats_resp.assert_status_is_ok();
    let stats: ClientStats = stats_resp.json().await.value().deserialize();
    assert_eq!(stats.completed_tasks, 1);
    assert_eq!(stats.failed_tasks, 1);
}

#[tokio::test]
async fn test_multiple_clients() {
    let server = Arc::new(HttpServer::new());
    let harness = TestHarness::new(server.clone(), create_routes(server));

    harness
        .server
        .enqueue_task_for_client(
            "client_A",
            TestTask {
                command: "echo".to_string(),
                data: "data_A".to_string(),
            },
        )
        .await
        .unwrap();
    harness
        .server
        .enqueue_task_for_client(
            "client_B",
            TestTask {
                command: "echo".to_string(),
                data: "data_B".to_string(),
            },
        )
        .await
        .unwrap();

    let resp_a = harness.client.get("/poll/client_A").send().await;
    resp_a.assert_status_is_ok();
    let task_a: TaskItem<TestTask> = resp_a.json().await.value().deserialize();
    assert_eq!(task_a.data.data, "data_A");

    let resp_b = harness.client.get("/poll/client_B").send().await;
    resp_b.assert_status_is_ok();
    let task_b: TaskItem<TestTask> = resp_b.json().await.value().deserialize();
    assert_eq!(task_b.data.data, "data_B");
}
