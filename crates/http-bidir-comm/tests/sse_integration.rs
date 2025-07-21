use std::sync::Arc;
use std::thread;
use std::time::Duration;

use async_stream::stream;
use http_bidir_comm::event_client::BlockingSseClient;
use http_bidir_comm::server::HttpServer;
use http_bidir_comm::ClientConfig;
use http_bidir_comm::TaskProcessor;
use http_bidir_comm::TaskResult;
use poem::get;
use poem::listener::TcpListener;
use poem::post;
use poem::web::sse::Event;
use poem::web::sse::SSE;
use poem::web::Data;
use poem::web::Json;
use poem::web::Path;
use poem::EndpointExt;
use poem::Route;
use poem::Server;
use serde::Deserialize;
use serde::Serialize;
use tokio::time::sleep as async_sleep;

type ServerType = HttpServer<TestTask, TestResult>;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestTask {
    value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestResult {
    value: String,
}

#[derive(Clone)]
struct EchoProcessor;

impl TaskProcessor<TestTask, TestResult> for EchoProcessor {
    fn process_task(
        &self,
        task: &TestTask,
    ) -> Result<TestResult, Box<dyn std::error::Error + Send + Sync>> {
        Ok(TestResult {
            value: task.value.to_uppercase(),
        })
    }
}

#[poem::handler]
async fn events_handler(
    Path(client_id): Path<String>,
    Data(server): Data<&Arc<ServerType>>,
) -> SSE {
    let server = Arc::clone(server);
    let sse_stream = stream! {
        loop {
            match server.poll_task_internal(&client_id).await {
                Ok(Some(task)) => {
                    if let Ok(json) = serde_json::to_string(&task) {
                        yield Event::message(json);
                    }
                }
                Ok(None) => {
                    async_sleep(Duration::from_millis(200)).await;
                }
                Err(err) => {
                    yield Event::message(err.to_string()).event_type("error");
                    break;
                }
            }
        }
    };

    SSE::new(sse_stream)
}

#[poem::handler]
async fn result_handler(
    Data(server): Data<&Arc<ServerType>>,
    Json(res): Json<TaskResult<TestResult>>,
) {
    let _ = server.submit_result_internal(res).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn end_to_end_sse() {
    // 1. start HttpServer
    let server = Arc::new(ServerType::new());

    // 2. build Poem routes
    let routes = Route::new()
        .at("/events/:client_id", get(events_handler))
        .at("/result", post(result_handler))
        .data(server.clone());

    // 3. bind random port
    let listener = TcpListener::bind("127.0.0.1:38041");
    let addr = "127.0.0.1:38041";

    // run server in background
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            Server::new(listener).run(routes).await.unwrap();
        });
    });

    // 4. enqueue task for client
    let client_id = "test_client".to_string();
    server
        .enqueue_task_for_client(
            &client_id,
            TestTask {
                value: "hello".into(),
            },
        )
        .await
        .unwrap();

    // 5. run BlockingSseClient in blocking thread
    let server_url = format!("http://{addr}");
    let cfg = ClientConfig::new(server_url.clone())
        .with_client_id(client_id.clone())
        .with_poll_interval(Duration::from_secs(1));

    let processor = Arc::new(EchoProcessor);

    let handle = thread::spawn(move || {
        // create client in blocking thread to avoid panic
        let sse_client = BlockingSseClient::<TestTask, TestResult>::new(cfg).unwrap();
        let _ = sse_client.start("", processor);
    });

    // 6. wait for processing
    async_sleep(Duration::from_secs(2)).await;

    // prevent dropping of runtime inside client thread to avoid panic
    std::mem::forget(handle);

    // 7. verify server stats
    let stats = server.get_client_stats(&client_id).await.unwrap();
    assert_eq!(stats.completed_tasks, 1);
}
