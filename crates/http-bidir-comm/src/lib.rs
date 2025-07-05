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
//! // Client usage
//! let config = ClientConfig::new("http://localhost:8080");
//! let client = BlockingHttpClient::<MyTask, MyResult>::new(config)?;
//!
//! // Server usage
//! let server = HttpServer::<MyTask, MyResult>::new();
//! let routes = create_routes(server.clone(), "/api/v1/tasks");
//! let app = Route::new().nest("/api/v1/tasks", routes);
//! let listener = poem::listener::TcpListener::bind("0.0.0.0:8080");
//! Server::new(listener).run(app).await.unwrap();
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

    #[derive(Clone)]
    #[expect(dead_code, reason = "Used in commented-out tests")]
    struct TestProcessor;

    impl TaskProcessor<TestTask, TestResult> for TestProcessor {
        fn process_task(
            &self,
            task: &TestTask,
        ) -> Result<TestResult, Box<dyn std::error::Error + Send + Sync>> {
            if task.command == "fail" {
                Err("Task failed".into())
            } else {
                Ok(TestResult {
                    id: task.id,
                    success: true,
                    message: Some(format!("Processed: {}", task.command)),
                })
            }
        }
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

    #[test(tokio::test)]
    async fn task_states_and_ids() {
        let task_id1 = TaskId::new();
        let task_id2 = TaskId::new();

        // Task IDs should be unique
        assert_ne!(task_id1, task_id2);

        // Task items should be created correctly
        let task = TestTask {
            id: 1,
            command: "test".to_string(),
        };
        let task_item = TaskItem::new(task.clone());

        assert_eq!(task_item.state, TaskState::Queued);
        assert_eq!(task_item.data, task);
        assert!(task_item.client_id.is_none());

        // Task results should be created correctly
        let success_result = TaskResult::success(task_id1, "client1".to_string(), TestResult {
            id: 1,
            success: true,
            message: Some("Done".to_string()),
        });

        assert!(success_result.success);
        assert!(success_result.result.is_some());
        assert!(success_result.error.is_none());

        let failure_result =
            TaskResult::<TestResult>::failure(task_id2, "client1".to_string(), "Error occurred");

        assert!(!failure_result.success);
        assert!(failure_result.result.is_none());
        assert!(failure_result.error.is_some());
    }

    #[test]
    fn client_creation() {
        let config = ClientConfig::new("http://localhost:8080");
        let client = BlockingHttpClient::<TestTask, TestResult>::new(config);

        assert!(client.is_ok());
    }

    #[test]
    fn blocking_client_creation() {
        let config = ClientConfig::new("http://localhost:8080");
        let client = BlockingHttpClient::<TestTask, TestResult>::new(config);

        assert!(client.is_ok());
    }
}
