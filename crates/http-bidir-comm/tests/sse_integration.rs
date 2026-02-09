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
use poem::http::StatusCode;
use poem::listener::TcpAcceptor;
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
use tracing::warn;

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
) -> StatusCode {
    match server.submit_result_internal(res).await {
        Ok(()) => StatusCode::OK,
        Err(err) => {
            warn!(error = %err, "Failed to submit task result");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn end_to_end_sse() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 1. start HttpServer
    let server = Arc::new(ServerType::new());

    // 2. build Poem routes
    let routes = Route::new()
        .at("/events/:client_id", get(events_handler))
        .at("/result", post(result_handler))
        .data(server.clone());

    // 3. bind random port
    let std_listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    std_listener.set_nonblocking(true)?;
    let local_addr = std_listener.local_addr()?;
    let acceptor = TcpAcceptor::from_std(std_listener)?;
    let addr = local_addr.to_string();

    // run server in background
    tokio::spawn(async move {
        if let Err(err) = Server::new_with_acceptor(acceptor).run(routes).await {
            warn!(error = %err, "Server run failed");
        }
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
        .await?;

    // 5. run BlockingSseClient in blocking thread
    let server_url = format!("http://{addr}");
    let cfg = ClientConfig::new(server_url.clone())
        .with_client_id(client_id.clone())
        .with_poll_interval(Duration::from_secs(1));

    let processor = Arc::new(EchoProcessor);

    let handle = thread::spawn(move || {
        let sse_client = match BlockingSseClient::<TestTask, TestResult>::new(cfg) {
            Ok(client) => client,
            Err(err) => {
                warn!(error = %err, "Failed to create blocking SSE client");
                return;
            }
        };

        if let Err(err) = sse_client.start("", processor) {
            warn!(error = %err, "Blocking SSE client exited");
        }
    });

    // 6. wait for processing
    let wait_deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(stats) = server.get_client_stats(&client_id).await {
            if stats.completed_tasks == 1 {
                break;
            }
        }

        if tokio::time::Instant::now() >= wait_deadline {
            break;
        }

        async_sleep(Duration::from_millis(100)).await;
    }

    // prevent dropping of runtime inside client thread to avoid panic
    std::mem::forget(handle);

    // 7. verify server stats
    let stats = server.get_client_stats(&client_id).await.ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "Missing client stats")
    })?;
    assert_eq!(stats.completed_tasks, 1);

    Ok(())
}
