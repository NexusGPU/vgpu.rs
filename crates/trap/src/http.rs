//! HTTP-based implementation of Trap communication
//!
//! This module provides HTTP API-based communication between trap clients and handlers,
//! using the generic `http_bidir_comm` library for bidirectional communication.
//!
//! The implementation maps trap concepts to generic task processing:
//! - Trap requests become tasks
//! - Trap responses become results
//! - Trap handlers implement the TaskProcessor trait

use std::io::Error as IoError;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use chrono::DateTime;
use chrono::Utc;
use http_bidir_comm::ClientConfig;
use http_bidir_comm::HttpServer;
use http_bidir_comm::ServerConfig;
use reqwest::blocking::Client as BlockingClient;
use reqwest::StatusCode;
use serde::Deserialize;
use serde::Serialize;
use uuid::Uuid;

use crate::Trap;
use crate::TrapAction;
use crate::TrapError;
use crate::TrapFrame;
use crate::TrapHandler;

/// HTTP-based trap request used as task data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpTrapRequest {
    pub trap_id: String,
    pub process_id: u32,
    pub frame: TrapFrame,
    pub timeout_seconds: u64,
}

/// HTTP-based trap response used as result data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpTrapResponse {
    pub trap_id: String,
    pub action: TrapAction,
    pub timestamp: DateTime<Utc>,
}

/// Configuration for HTTP trap client
#[derive(Debug, Clone)]
pub struct HttpTrapConfig {
    pub server_url: String,
    pub timeout: Duration,
    pub retry_attempts: u32,
    pub retry_delay: Duration,
    pub client_id: String,
    pub base_path: String,
}

impl Default for HttpTrapConfig {
    fn default() -> Self {
        Self {
            server_url: "http://localhost:8080".to_string(),
            timeout: Duration::from_secs(30),
            retry_attempts: 3,
            retry_delay: Duration::from_millis(500),
            client_id: format!("trap_client_{}", std::process::id()),
            base_path: "/api/v1/trap".to_string(),
        }
    }
}

impl From<HttpTrapConfig> for ClientConfig {
    fn from(config: HttpTrapConfig) -> Self {
        ClientConfig::new(config.server_url)
            .with_client_id(config.client_id)
            .with_request_timeout(config.timeout)
            .with_retry_config(
                config.retry_attempts,
                config.retry_delay,
                config.retry_delay * 10,
            )
    }
}
/// Blocking HTTP trap client implementation
pub struct BlockingHttpTrap {
    config: HttpTrapConfig,
}

impl BlockingHttpTrap {
    /// Create a new blocking HTTP trap client.
    pub fn new(config: HttpTrapConfig) -> Result<Self, TrapError> {
        // Validate configuration by building a simple reqwest client; this ensures
        // obvious mis-configurations are caught early without storing an unused field.
        BlockingClient::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| {
                TrapError::Io(IoError::other(format!("Failed to create HTTP client: {e}")))
            })?;

        Ok(Self { config })
    }

    /// Send a trap request and wait for the handler's response (synchronous HTTP).
    pub fn send_trap_request(&self, frame: TrapFrame) -> Result<TrapAction, TrapError> {
        // Compose basic request information.
        let trap_id = format!("{}_{}", self.config.client_id, Uuid::new_v4());
        let process_id = std::process::id();

        let request = HttpTrapRequest {
            trap_id,
            process_id,
            frame,
            timeout_seconds: self.config.timeout.as_secs(),
        };

        // Construct the target URL. The API follows the pattern
        //   {server_url}{base_path}
        // where `base_path` defaults to "/api/v1/trap".
        let url = format!("{}{}", self.config.server_url, self.config.base_path);

        // Build a dedicated blocking client with the configured timeout so we don't rely on
        // internal details of `http_bidir_comm::BlockingHttpClient`.
        let http_client = BlockingClient::builder()
            .timeout(self.config.timeout)
            .build()
            .map_err(|e| TrapError::Io(IoError::other(e.to_string())))?;

        // Perform the POST request with retries if configured.
        // A very small retry helper – we keep it simple for now.
        let mut attempts = 0;
        loop {
            attempts += 1;
            match http_client.post(&url).json(&request).send() {
                Ok(resp) => {
                    if resp.status().is_success() {
                        // Successful response – parse JSON body.
                        let trap_resp: HttpTrapResponse = resp.json().map_err(|e| {
                            TrapError::Io(IoError::other(format!(
                                "Failed to deserialize trap response: {e}"
                            )))
                        })?;
                        return Ok(trap_resp.action);
                    } else {
                        // Non-200 response -> treat as error.
                        let status = resp.status();
                        let text = resp.text().unwrap_or_default();
                        if attempts <= self.config.retry_attempts {
                            std::thread::sleep(self.config.retry_delay);
                            continue;
                        }
                        return Err(TrapError::Io(IoError::other(format!(
                            "Trap server responded with {status}: {text}"
                        ))));
                    }
                }
                Err(err) => {
                    // If we have retries left, wait a bit and retry; otherwise propagate.
                    if attempts <= self.config.retry_attempts {
                        std::thread::sleep(self.config.retry_delay);
                        continue;
                    }
                    return Err(TrapError::Io(IoError::other(format!(
                        "HTTP request failed: {err}"
                    ))));
                }
            }
        }
    }
}

impl Trap for BlockingHttpTrap {
    fn enter_trap_and_wait(&self, frame: TrapFrame) -> Result<TrapAction, TrapError> {
        self.send_trap_request(frame)
    }
}

/// HTTP-based trap server using http_bidir_comm
pub struct HttpTrapServer<H: TrapHandler + Send + Sync + Clone + 'static> {
    server: HttpServer<HttpTrapRequest, HttpTrapResponse>,
    _phantom: PhantomData<Arc<H>>,
}

impl<H: TrapHandler + Send + Sync + Clone + 'static> HttpTrapServer<H> {
    pub fn new(handler: H, config: Option<ServerConfig>) -> Self {
        let server = match config {
            Some(cfg) => HttpServer::with_config(cfg),
            None => HttpServer::new(),
        };

        let _ = handler; // handler is currently unused but kept for future extensions
        Self {
            server,
            _phantom: PhantomData,
        }
    }

    /// Get the underlying server for route creation
    ///
    /// Note: The http_bidir_comm server is passive - it doesn't have a start_processing method.
    /// Instead, you need to integrate it with a web framework like Poem using routes.
    pub fn get_server(&self) -> &HttpServer<HttpTrapRequest, HttpTrapResponse> {
        &self.server
    }

    /// Enqueue a trap request for processing
    pub async fn enqueue_trap_request(&self, request: HttpTrapRequest) -> Result<(), TrapError> {
        let client_id = request.trap_id.clone();
        self.server
            .enqueue_task_for_client(&client_id, request)
            .await
            .map_err(|e| {
                TrapError::Io(IoError::other(format!(
                    "Failed to enqueue trap request: {e}"
                )))
            })?;
        Ok(())
    }

    /// Get server statistics
    pub async fn get_stats(&self) -> Result<serde_json::Value, TrapError> {
        let stats = self.server.get_all_stats().await;
        serde_json::to_value(stats)
            .map_err(|e| TrapError::Io(IoError::other(format!("Failed to serialize stats: {e}"))))
    }
}

/// Helper function to create trap routes for integration into existing servers
pub fn create_trap_server<H: TrapHandler + Send + Sync + Clone + 'static>(
    handler: H,
    config: Option<ServerConfig>,
) -> HttpTrapServer<H> {
    HttpTrapServer::new(handler, config)
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener as StdTcpListener;
    use std::sync::Arc;
    use std::sync::Mutex;

    use http_bidir_comm::TaskProcessor;
    use poem::listener::TcpListener;
    use poem::post;
    use poem::web::Data;
    use poem::web::Json;
    use poem::EndpointExt;
    use poem::Route;
    use poem::Server;
    use tokio::task;

    use super::*;
    use crate::Waker;

    /// Task processor that handles trap requests using the TrapHandler
    #[derive(Clone)]
    struct TrapTaskProcessor<H: TrapHandler + Send + Sync + Clone> {
        handler: Arc<H>,
    }

    impl<H: TrapHandler + Send + Sync + Clone> TaskProcessor<HttpTrapRequest, HttpTrapResponse>
        for TrapTaskProcessor<H>
    {
        fn process_task(
            &self,
            task: &HttpTrapRequest,
        ) -> Result<HttpTrapResponse, Box<dyn std::error::Error + Send + Sync>> {
            // Extract trap ID as u64 for the handler
            let trap_id = task.trap_id.parse::<u64>().unwrap_or(0);

            // Create a simple waker that captures the response
            let waker = SimpleWaker::new();

            // Handle the trap
            self.handler.handle_trap(
                task.process_id,
                trap_id,
                &task.frame,
                Box::new(waker.clone()),
            );

            // Get the action from the waker
            let action = waker.get_action().unwrap_or(TrapAction::Resume);

            Ok(HttpTrapResponse {
                trap_id: task.trap_id.clone(),
                action,
                timestamp: Utc::now(),
            })
        }
    }

    /// Simple waker implementation for capturing trap actions
    #[derive(Clone)]
    struct SimpleWaker {
        action: Arc<std::sync::Mutex<Option<TrapAction>>>,
    }

    impl SimpleWaker {
        fn new() -> Self {
            Self {
                action: Arc::new(std::sync::Mutex::new(None)),
            }
        }

        fn get_action(&self) -> Option<TrapAction> {
            self.action.lock().unwrap().clone()
        }
    }

    impl Waker for SimpleWaker {
        fn send(&self, _trap_id: u64, action: TrapAction) -> Result<(), TrapError> {
            let mut action_guard = self.action.lock().unwrap();
            *action_guard = Some(action);
            Ok(())
        }
    }

    /// A trap handler that records trap requests for testing
    #[derive(Clone)]
    struct RecordingTrapHandler {
        requests: Arc<Mutex<Vec<(u32, u64, TrapFrame)>>>,
    }

    impl RecordingTrapHandler {
        fn new() -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn get_requests(&self) -> Vec<(u32, u64, TrapFrame)> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl TrapHandler for RecordingTrapHandler {
        fn handle_trap(&self, pid: u32, trap_id: u64, frame: &TrapFrame, waker: Box<dyn Waker>) {
            self.requests
                .lock()
                .unwrap()
                .push((pid, trap_id, frame.clone()));
            let _ = waker.send(trap_id, TrapAction::Resume);
        }
    }

    /// test TrapHandler: call waker and always return Resume
    #[derive(Clone)]
    struct TestTrapHandler;

    impl TrapHandler for TestTrapHandler {
        fn handle_trap(&self, _pid: u32, trap_id: u64, _frame: &TrapFrame, waker: Box<dyn Waker>) {
            let _ = waker.send(trap_id, TrapAction::Resume);
        }
    }

    // use poem::handler to create a endpoint that can be used in poem::Route
    #[poem::handler]
    async fn trap_endpoint(
        Json(req): Json<HttpTrapRequest>,
        Data(handler): Data<&Arc<TestTrapHandler>>,
    ) -> Json<HttpTrapResponse> {
        let waker = SimpleWaker::new();
        handler.handle_trap(
            req.process_id,
            req.trap_id.parse::<u64>().unwrap_or(0),
            &req.frame,
            Box::new(waker.clone()),
        );
        let action = waker.get_action().unwrap_or(TrapAction::Resume);
        Json(HttpTrapResponse {
            trap_id: req.trap_id.clone(),
            action,
            timestamp: Utc::now(),
        })
    }

    #[tokio::test]
    async fn end_to_end_trap_flow() {
        // create a shared handler
        let handler = Arc::new(TestTrapHandler);

        // build poem route and inject handler data
        let route = Route::new()
            .at("/api/v1/trap", post(trap_endpoint))
            .data(handler.clone());

        // create std listener bind random port, get port and release, then use poem bind same port
        let std_listener = StdTcpListener::bind("127.0.0.1:0").expect("bind std listener");
        let local_addr = std_listener.local_addr().expect("local addr");
        let port = local_addr.port();
        drop(std_listener); // release port

        let poem_listener = TcpListener::bind(format!("127.0.0.1:{port}"));
        let server_url = format!("http://127.0.0.1:{port}");

        // run poem server in background
        let server_handle = tokio::spawn(async move {
            Server::new(poem_listener).run(route).await.unwrap();
        });

        // The test was failing because the blocking reqwest client (and its internal Tokio runtime)
        // was created and dropped within an async context. To fix this, we move the client
        // creation and the blocking call into a separate thread using `task::spawn_blocking`.
        let trap_task = task::spawn_blocking(move || {
            // build TrapClient to test server
            let config = HttpTrapConfig {
                server_url,
                ..Default::default()
            };
            let trap_client = BlockingHttpTrap::new(config).expect("create client");

            // prepare trap frame
            let frame = TrapFrame::OutOfMemory {
                requested_bytes: 128,
            };

            // call enter_trap_and_wait
            trap_client.enter_trap_and_wait(frame)
        });

        let action = trap_task.await.expect("join").expect("trap ok");

        assert!(matches!(action, TrapAction::Resume));

        // stop server
        server_handle.abort();
    }

    #[tokio::test]
    async fn test_trap_task_processor() {
        let handler = RecordingTrapHandler::new();
        let processor = TrapTaskProcessor {
            handler: Arc::new(handler.clone()),
        };

        let request = HttpTrapRequest {
            trap_id: "test_trap_456".to_string(),
            process_id: 5678,
            frame: TrapFrame::OutOfMemory {
                requested_bytes: 2048,
            },
            timeout_seconds: 60,
        };

        let result = processor.process_task(&request);
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.trap_id, "test_trap_456");
        assert!(matches!(response.action, TrapAction::Resume));

        // Verify the handler was called
        let requests = handler.get_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].0, 5678); // process_id
        assert_eq!(requests[0].1, 0); // trap_id (parsed from string, defaults to 0)
    }

    #[test]
    fn test_config_conversion() {
        let trap_config = HttpTrapConfig {
            server_url: "http://test.example.com".to_string(),
            timeout: Duration::from_secs(60),
            retry_attempts: 5,
            retry_delay: Duration::from_millis(200),
            client_id: "test_client".to_string(),
            base_path: "/test/api".to_string(),
        };

        let client_config = ClientConfig::from(trap_config.clone());
        assert_eq!(client_config.server_url, "http://test.example.com");
        assert_eq!(client_config.client_id, "test_client");
        assert_eq!(client_config.request_timeout, Duration::from_secs(60));
        assert_eq!(client_config.max_retries, 5);
    }

    #[tokio::test]
    async fn test_blocking_http_trap_retry_mechanism() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // Start a mock server.
        let mock_server = MockServer::start().await;
        let uri = mock_server.uri();

        // The server will fail twice with 503, then succeed once with 200.
        Mock::given(method("POST"))
            .and(path("/api/v1/trap"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(2)
            .mount(&mock_server)
            .await;

        let success_response = HttpTrapResponse {
            trap_id: "some_id".to_string(),
            action: TrapAction::Resume,
            timestamp: Utc::now(),
        };

        Mock::given(method("POST"))
            .and(path("/api/v1/trap"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&success_response))
            .mount(&mock_server)
            .await;

        // Configure the client to retry 3 times.
        let config = HttpTrapConfig {
            server_url: uri,
            retry_attempts: 3,
            retry_delay: Duration::from_millis(100),
            ..Default::default()
        };

                let result = tokio::task::spawn_blocking(move || {
            let trap_client = BlockingHttpTrap::new(config).unwrap();
            let frame = TrapFrame::OutOfMemory { requested_bytes: 128 };
            trap_client.send_trap_request(frame)
        })
        .await
        .unwrap();
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), TrapAction::Resume));

        // Verify that the server was called 3 times (2 failures + 1 success).
        let received_requests = mock_server.received_requests().await.unwrap();
        assert_eq!(received_requests.len(), 3);
    }

    #[tokio::test]
    async fn test_blocking_http_trap_handles_error_status() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // Start a mock server.
        let mock_server = MockServer::start().await;
        let uri = mock_server.uri();

        // The server will consistently return a 400 Bad Request.
        Mock::given(method("POST"))
            .and(path("/api/v1/trap"))
            .respond_with(ResponseTemplate::new(400).set_body_string("Invalid request format"))
            .mount(&mock_server)
            .await;

        let config = HttpTrapConfig {
            server_url: uri,
            retry_attempts: 1, // No need to retry for this test
            ..Default::default()
        };

                        let result = tokio::task::spawn_blocking(move || {
            let trap_client = BlockingHttpTrap::new(config).unwrap();
            let frame = TrapFrame::OutOfMemory { requested_bytes: 128 };
            trap_client.send_trap_request(frame)
        })
        .await
        .unwrap();
        assert!(result.is_err());

        if let Err(TrapError::Io(e)) = result {
            let error_string = e.to_string();
            assert!(error_string.contains("400 Bad Request"));
            assert!(error_string.contains("Invalid request format"));
        } else {
            panic!("Expected TrapError::Io");
        }
    }

    #[tokio::test]
    async fn test_blocking_http_trap_request_timeout() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // Start a mock server.
        let mock_server = MockServer::start().await;
        let uri = mock_server.uri();

        // The server will delay its response longer than the client's timeout.
        Mock::given(method("POST"))
            .and(path("/api/v1/trap"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(2)))
            .mount(&mock_server)
            .await;

        let config = HttpTrapConfig {
            server_url: uri,
            timeout: Duration::from_secs(1),
            retry_attempts: 0, // Disable retries for a clean timeout test
            ..Default::default()
        };

                        let result = tokio::task::spawn_blocking(move || {
            let trap_client = BlockingHttpTrap::new(config).unwrap();
            let frame = TrapFrame::OutOfMemory { requested_bytes: 128 };
            trap_client.send_trap_request(frame)
        })
        .await
        .unwrap();
        assert!(result.is_err());

        if let Err(TrapError::Io(e)) = result {
            let error_string = e.to_string();
                                                assert!(error_string.contains("HTTP request failed"));
        } else {
            panic!("Expected TrapError::Io for timeout");
        }
    }
}
