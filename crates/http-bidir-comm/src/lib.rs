//! HTTP-based bidirectional communication library.
//!
//! This library provides a generic framework for HTTP-based bidirectional communication
//! between clients and servers. It abstracts the common pattern where:
//!
//! - Clients poll the server for tasks/commands
//! - Clients submit results/responses back to the server
//! - Server manages queues and handles requests
//!
//! # Features
//!
//! - Generic task and result types
//! - Automatic reconnection and retry logic
//! - Integration with Poem web framework
//! - Configurable endpoints and timeouts
//! - Structured error handling with context
//!
//! # Examples
//!
//! ```
//! # use http_bidir_comm::{ClientConfig, BlockingHttpClient, HttpServer, ServerConfig};
//! # use http_bidir_comm::poem::create_routes;
//! # use poem::{Route, Server};
//! # use std::sync::Arc;
//! # use serde::{Deserialize, Serialize};
//! #
//! # #[derive(Debug, Clone, Serialize, Deserialize)]
//! # struct MyTask {
//! #     id: u64,
//! #     command: String,
//! # }
//! #
//! # #[derive(Debug, Clone, Serialize, Deserialize)]
//! # struct MyResult {
//! #     id: u64,
//! #     success: bool,
//! #     message: Option<String>,
//! # }
//! #
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Client usage - BlockingHttpClient should be used in non-async context
//! let config = ClientConfig::new("http://localhost:8080");
//! let client = BlockingHttpClient::<MyTask, MyResult>::new(config)?;
//!
//! // Server usage - can be used in async context
//! let rt = tokio::runtime::Runtime::new()?;
//! rt.block_on(async {
//!     let server = Arc::new(HttpServer::<MyTask, MyResult>::new());
//!     let routes = create_routes(server.clone(), "/api/v1/tasks");
//!     let app = Route::new().nest("/api/v1/tasks", routes);
//!     let listener = poem::listener::TcpListener::bind("0.0.0.0:8080");
//!     // Server::new(listener).run(app).await.unwrap();
//! });
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod error;
pub mod event_client;
pub mod poem;
pub mod server;
pub mod types;

pub use config::ClientConfig;
pub use error::CommError;
pub use error::CommResult;
pub use event_client::BlockingSseClient as BlockingHttpClient;
pub use poem::Route;
pub use server::ClientStats;
pub use server::HttpServer;
pub use server::ServerConfig;
pub use types::TaskId;
pub use types::TaskItem;
pub use types::TaskProcessor;
pub use types::TaskResult;
pub use types::TaskState;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde::Deserialize;
    use serde::Serialize;
    use similar_asserts::assert_eq;
    use test_log::test;

    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestTask {
        id: u64,
        command: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestResult {
        id: u64,
        success: bool,
        message: Option<String>,
    }

    #[test(tokio::test)]
    async fn create_server_and_enqueue_task() {
        let server = HttpServer::<TestTask, TestResult>::new();

        let task = TestTask {
            id: 1,
            command: "test".to_string(),
        };

        let _task_id = server
            .enqueue_task_for_client("default", task)
            .await
            .expect("should enqueue task");

        // Verify task was enqueued
        let stats = server.get_all_stats().await;
        assert_eq!(stats.len(), 1);
        assert_eq!(
            stats
                .get("default")
                .expect("should have default client")
                .pending_tasks,
            1
        );
    }

    #[test(tokio::test)]
    async fn create_client_config() {
        let config = ClientConfig::new("http://localhost:8080")
            .with_client_id("test_client")
            .with_poll_interval(Duration::from_millis(500))
            .with_request_timeout(Duration::from_secs(10));

        assert_eq!(config.server_url, "http://localhost:8080");
        assert_eq!(config.client_id, "test_client");
        assert_eq!(config.poll_interval, Duration::from_millis(500));
        assert_eq!(config.request_timeout, Duration::from_secs(10));
    }

    #[test]
    fn blocking_client_creation() {
        let config = ClientConfig::new("http://localhost:8080");
        let client = BlockingHttpClient::<TestTask, TestResult>::new(config);

        assert!(client.is_ok());
    }
}
