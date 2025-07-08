use std::sync::Arc;
use std::time::Duration;

use http_bidir_comm::error::CommError;
use http_bidir_comm::server::HttpServer;
use http_bidir_comm::server::ServerConfig;
use http_bidir_comm::types::TaskProcessor;
use http_bidir_comm::ClientConfig;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct SimpleTask {
    id: u32,
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct SimpleResult {
    task_id: u32,
    processed_message: String,
}

struct SimpleProcessor;

impl TaskProcessor<SimpleTask, SimpleResult> for SimpleProcessor {
    fn process_task(
        &self,
        task: &SimpleTask,
    ) -> Result<SimpleResult, Box<dyn std::error::Error + Send + Sync>> {
        Ok(SimpleResult {
            task_id: task.id,
            processed_message: format!("Processed: {}", task.message),
        })
    }
}

#[tokio::test]
async fn server_creation_and_basic_operations() {
    let server = HttpServer::<SimpleTask, SimpleResult>::new();

    // Test basic server creation
    let stats = server.get_all_stats().await;
    assert!(stats.is_empty(), "Should have no client stats initially");

    // Test client-specific stats
    let client_stats = server.get_client_stats("test_client").await;
    assert!(
        client_stats.is_none(),
        "Should have no stats for non-existent client"
    );
}

#[tokio::test]
async fn client_config_creation() {
    let config = ClientConfig::new("http://localhost:8080")
        .with_client_id("test_client")
        .with_request_timeout(Duration::from_secs(5))
        .with_poll_interval(Duration::from_millis(100))
        .with_retry_config(3, Duration::from_millis(100), Duration::from_secs(10));

    assert_eq!(config.server_url, "http://localhost:8080");
    assert_eq!(config.client_id, "test_client");
    assert_eq!(config.request_timeout, Duration::from_secs(5));
    assert_eq!(config.poll_interval, Duration::from_millis(100));
    assert_eq!(config.max_retries, 3);
}

#[tokio::test]
async fn server_config_creation() {
    let config = ServerConfig::default();
    assert_eq!(config.max_queue_size, 1000);
    assert_eq!(config.max_history_size, 100);
    assert!(!config.enable_persistence);

    let server = HttpServer::<SimpleTask, SimpleResult>::with_config(config.clone());
    let stats = server.get_all_stats().await;
    assert!(stats.is_empty(), "New server should have no client stats");
}

#[tokio::test]
async fn task_serialization() {
    let task = SimpleTask {
        id: 42,
        message: "hello world".to_string(),
    };
    let serialized = serde_json::to_string(&task).expect("should serialize task");
    let deserialized: SimpleTask =
        serde_json::from_str(&serialized).expect("should deserialize task");

    assert_eq!(
        task, deserialized,
        "Task should serialize/deserialize correctly"
    );
}

#[tokio::test]
async fn result_serialization() {
    let result = SimpleResult {
        task_id: 42,
        processed_message: "processed".to_string(),
    };
    let serialized = serde_json::to_string(&result).expect("should serialize result");
    let deserialized: SimpleResult =
        serde_json::from_str(&serialized).expect("should deserialize result");

    assert_eq!(
        result, deserialized,
        "Result should serialize/deserialize correctly"
    );
}

#[tokio::test]
async fn processor_functionality() {
    let processor = SimpleProcessor;
    let task = SimpleTask {
        id: 1,
        message: "test message".to_string(),
    };

    let result = processor
        .process_task(&task)
        .expect("should process task successfully");
    assert_eq!(result.task_id, 1, "Result should have correct task ID");
    assert_eq!(
        result.processed_message, "Processed: test message",
        "Result should have processed message"
    );
}

#[tokio::test]
async fn error_handling() {
    // Test error creation
    let error = CommError::Configuration {
        message: "test error".into(),
    };
    assert!(
        error.to_string().contains("test error"),
        "Error should contain message"
    );

    // Test network error
    let network_error = CommError::Network {
        message: "network failure".into(),
    };
    assert!(
        network_error.to_string().contains("network failure"),
        "Network error should contain message"
    );

    // Test HTTP error
    let http_error = CommError::Http {
        status: 404,
        message: "Not Found".into(),
    };
    assert!(
        http_error.to_string().contains("404"),
        "HTTP error should contain status code"
    );
}

#[tokio::test]
async fn task_queue_operations() {
    let server = HttpServer::<SimpleTask, SimpleResult>::new();

    // Test enqueue for a specific client
    let task = SimpleTask {
        id: 1,
        message: "test".to_string(),
    };
    let _task_id = server
        .enqueue_task_for_client("test_client", task)
        .await
        .expect("should enqueue task");

    // Check stats after enqueue
    let stats = server.get_all_stats().await;
    assert_eq!(stats.len(), 1, "Should have one client with stats");
    assert!(
        stats.contains_key("test_client"),
        "Should have test_client in stats"
    );

    let client_stats = stats.get("test_client").unwrap();
    assert_eq!(
        client_stats.pending_tasks, 1,
        "Should have one pending task"
    );
    assert_eq!(
        client_stats.processing_tasks, 0,
        "Should have no processing tasks"
    );
    assert_eq!(
        client_stats.completed_tasks, 0,
        "Should have no completed tasks"
    );
}

#[tokio::test]
async fn large_task_data() {
    let server = HttpServer::<SimpleTask, SimpleResult>::new();

    // Create a task with large data
    let large_message = "x".repeat(10000);
    let task = SimpleTask {
        id: 1,
        message: large_message.clone(),
    };

    let result = server.enqueue_task_for_client("test_client", task).await;
    assert!(result.is_ok(), "Should handle large task data");

    // Test serialization of large data
    let task = SimpleTask {
        id: 1,
        message: large_message,
    };
    let serialized = serde_json::to_string(&task).expect("should serialize large task");
    assert!(serialized.len() > 10000, "Serialized data should be large");
}

#[tokio::test]
async fn special_characters_in_data() {
    let server = HttpServer::<SimpleTask, SimpleResult>::new();

    // Test with special characters
    let special_message = "Hello 世界! 🦀 Rust \"quotes\" and\nnewlines\ttabs";
    let task = SimpleTask {
        id: 1,
        message: special_message.to_string(),
    };

    let result = server
        .enqueue_task_for_client("test_client", task.clone())
        .await;
    assert!(
        result.is_ok(),
        "Should handle special characters in task data"
    );

    // Test serialization with special characters
    let serialized =
        serde_json::to_string(&task).expect("should serialize task with special chars");
    let deserialized: SimpleTask =
        serde_json::from_str(&serialized).expect("should deserialize task with special chars");

    assert_eq!(
        task, deserialized,
        "Task with special characters should serialize/deserialize correctly"
    );
}

#[tokio::test]
async fn concurrent_task_enqueuing() {
    let server = Arc::new(HttpServer::<SimpleTask, SimpleResult>::new());

    let mut handles = Vec::new();

    for i in 0..10 {
        let server = server.clone();
        let handle = tokio::spawn(async move {
            let task = SimpleTask {
                id: i,
                message: format!("task {i}"),
            };
            server
                .enqueue_task_for_client("concurrent_client", task)
                .await
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    let stats = server.get_client_stats("concurrent_client").await.unwrap();
    assert_eq!(stats.pending_tasks, 10, "Should have 10 pending tasks");
}

#[tokio::test]
async fn multiple_clients_different_tasks() {
    let server = Arc::new(HttpServer::<SimpleTask, SimpleResult>::new());

    let server_a = server.clone();
    let handle_a = tokio::spawn(async move {
        for i in 0..5 {
            let task = SimpleTask {
                id: i,
                message: format!("client_a_task_{i}"),
            };
            server_a
                .enqueue_task_for_client("client_a", task)
                .await
                .unwrap();
        }
    });

    let server_b = server.clone();
    let handle_b = tokio::spawn(async move {
        for i in 0..5 {
            let task = SimpleTask {
                id: i,
                message: format!("client_b_task_{i}"),
            };
            server_b
                .enqueue_task_for_client("client_b", task)
                .await
                .unwrap();
        }
    });

    handle_a.await.unwrap();
    handle_b.await.unwrap();

    let stats_a = server.get_client_stats("client_a").await.unwrap();
    assert_eq!(stats_a.pending_tasks, 5, "Client A should have 5 tasks");

    let stats_b = server.get_client_stats("client_b").await.unwrap();
    assert_eq!(stats_b.pending_tasks, 5, "Client B should have 5 tasks");
}
