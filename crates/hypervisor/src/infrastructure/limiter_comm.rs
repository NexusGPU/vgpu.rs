//! limiter communication using the http-bidir-comm library.
//!
//! This module demonstrates how to replace the existing limiter_comm implementation
//! with the new generic HTTP bidirectional communication library.

use std::collections::HashMap;
use std::sync::Arc;

use error_stack::Report;
use http_bidir_comm::poem;
use http_bidir_comm::{ClientStats, CommError, HttpServer, ServerConfig};
use tracing::info;
use tracing::instrument;

use crate::api::LimiterCommand;
use crate::api::LimiterCommandResponse;
use crate::api::LimiterCommandType;

/// Command dispatcher using the new HTTP bidirectional communication library.
pub struct CommandDispatcher {
    server: Arc<HttpServer<LimiterCommand, LimiterCommandResponse>>,
}

impl CommandDispatcher {
    /// Create a new command dispatcher.
    pub fn new() -> Self {
        let config = ServerConfig {
            max_queue_size: 1000,
            max_history_size: 100,
            enable_persistence: false,
        };

        let server = Arc::new(HttpServer::with_config(config));

        info!("command dispatcher created");

        Self { server }
    }

    /// Enqueue a command for a specific limiter.
    ///
    /// # Arguments
    ///
    /// * `limiter_id` - The ID of the target limiter
    /// * `kind` - The type of command to enqueue
    ///
    /// # Errors
    ///
    /// Returns an error if the command cannot be enqueued
    #[instrument(skip(self))]
    pub async fn enqueue_command(
        &self,
        limiter_id: &str,
        kind: LimiterCommandType,
    ) -> Result<u64, Report<CommError>> {
        // Create a command with a unique ID
        let id = generate_command_id();
        let cmd = LimiterCommand { id, kind };

        // Enqueue the command for the specific limiter
        let task_id = self.server.enqueue_task_for_client(limiter_id, cmd).await?;

        info!(limiter_id = %limiter_id, command_id = id, task_id = %task_id, "Command enqueued");

        Ok(id)
    }

    /// Get statistics for all limiter clients.
    #[allow(dead_code)]
    pub async fn get_all_stats(&self) -> HashMap<String, ClientStats> {
        self.server.get_all_stats().await
    }

    /// Get statistics for a specific limiter client.
    #[allow(dead_code)]
    pub async fn get_client_stats(&self, limiter_id: &str) -> Option<ClientStats> {
        self.server.get_client_stats(limiter_id).await
    }

    /// Create Poem routes for the limiter communication API.
    pub fn create_routes(&self) -> poem::Route {
        self.create_routes_with_path("")
    }

    /// Create routes with a custom base path.
    pub fn create_routes_with_path(&self, base_path: &str) -> poem::Route {
        poem::create_routes(self.server.clone(), base_path)
    }
}

impl Default for CommandDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a unique command ID.
fn generate_command_id() -> u64 {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use similar_asserts::assert_eq;
    use test_log::test;

    use super::*;

    #[test(tokio::test)]
    async fn create_dispatcher() {
        let dispatcher = CommandDispatcher::new();

        // Should be able to get stats for non-existent clients
        let stats = dispatcher.get_all_stats().await;
        assert_eq!(stats.len(), 0);

        let client_stats = dispatcher.get_client_stats("non_existent").await;
        assert!(client_stats.is_none());
    }

    #[test(tokio::test)]
    async fn enqueue_command() {
        let dispatcher = CommandDispatcher::new();

        let command_id = dispatcher
            .enqueue_command("test_limiter", LimiterCommandType::TfHealthCheck)
            .await
            .expect("should enqueue command");

        assert!(command_id > 0);

        // Check that the limiter client now has pending tasks
        let stats = dispatcher.get_client_stats("test_limiter").await;
        assert!(stats.is_some());

        let client_stats = stats.expect("should have client stats");
        assert_eq!(client_stats.pending_tasks, 1);
    }

    #[test(tokio::test)]
    async fn create_routes() {
        let dispatcher = CommandDispatcher::new();
        let routes = dispatcher.create_routes();

        // Routes should be created successfully
        let _ = routes;
    }

    #[test(tokio::test)]
    async fn queue_overflow() {
        let dispatcher = CommandDispatcher::new();

        // fill queue to the configured limit (1000)
        for _ in 0..1000 {
            dispatcher
                .enqueue_command("overflow_limiter", LimiterCommandType::TfHealthCheck)
                .await
                .expect("should enqueue until limit");
        }

        // the 1001st enqueue should fail
        let result = dispatcher
            .enqueue_command("overflow_limiter", LimiterCommandType::TfHealthCheck)
            .await;
        assert!(result.is_err(), "should fail when queue is full");

        // queue stats should stay at the limit value
        let stats = dispatcher
            .get_client_stats("overflow_limiter")
            .await
            .expect("should have stats");
        assert_eq!(stats.pending_tasks, 1000);
    }

    #[test(tokio::test)]
    async fn multiple_limiters_stats() {
        let dispatcher = CommandDispatcher::new();

        // enqueue different number of commands for multiple limiters
        dispatcher
            .enqueue_command("limiter1", LimiterCommandType::TfHealthCheck)
            .await
            .expect("should enqueue command");
        dispatcher
            .enqueue_command("limiter1", LimiterCommandType::TfResume)
            .await
            .expect("should enqueue second command");
        dispatcher
            .enqueue_command("limiter2", LimiterCommandType::TfSuspend)
            .await
            .expect("should enqueue command for second limiter");

        // verify each limiter's queue stats
        let stats1 = dispatcher
            .get_client_stats("limiter1")
            .await
            .expect("should have stats for limiter1");
        assert_eq!(stats1.pending_tasks, 2);

        let stats2 = dispatcher
            .get_client_stats("limiter2")
            .await
            .expect("should have stats for limiter2");
        assert_eq!(stats2.pending_tasks, 1);

        // verify global stats
        let all_stats = dispatcher.get_all_stats().await;
        assert_eq!(all_stats.len(), 2);
    }
}
