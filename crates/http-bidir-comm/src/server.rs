//! HTTP server for bidirectional communication.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;

use chrono::Utc;
use error_stack::bail;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::RwLock;
use tracing::debug;
use tracing::info;
use tracing::instrument;
use tracing::trace;
use tracing::warn;
use std::borrow::Cow;

use crate::error::CommError;
use crate::error::CommResult;
use crate::types::TaskId;
use crate::types::TaskItem;
use crate::types::TaskResult;
use crate::types::TaskState;

/// Configuration for HTTP server.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Maximum number of tasks to keep in queue per client
    pub max_queue_size: usize,
    /// Maximum number of completed tasks to keep in history
    pub max_history_size: usize,
    /// Whether to enable task persistence
    pub enable_persistence: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            max_history_size: 100,
            enable_persistence: false,
        }
    }
}

/// Task queue manager for a single client.
#[derive(Debug)]
struct ClientQueue<T, R>
where T: Clone
{
    /// Tasks waiting to be processed
    pending: VecDeque<TaskItem<T>>,
    /// Tasks currently being processed
    processing: HashMap<TaskId, TaskItem<T>>,
    /// Completed tasks history
    completed: VecDeque<TaskResult<R>>,
    /// Last activity timestamp
    last_activity: chrono::DateTime<Utc>,
}

impl<T, R> ClientQueue<T, R>
where T: Clone
{
    fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            processing: HashMap::new(),
            completed: VecDeque::new(),
            last_activity: Utc::now(),
        }
    }

    fn enqueue_task(&mut self, task: TaskItem<T>, max_queue_size: usize) -> CommResult<()> {
        if self.pending.len() >= max_queue_size {
            bail!(CommError::ServerState {
                message: Cow::Borrowed("Task queue is full"),
            });
        }

        self.pending.push_back(task);
        self.last_activity = Utc::now();
        Ok(())
    }

    fn dequeue_task(&mut self, client_id: &str) -> Option<TaskItem<T>> {
        if let Some(mut task) = self.pending.pop_front() {
            task.state = TaskState::Processing;
            task.client_id = Some(client_id.to_string());
            task.updated_at = Utc::now();

            self.processing.insert(task.id, task.clone());
            self.last_activity = Utc::now();

            Some(task)
        } else {
            None
        }
    }

    fn complete_task(&mut self, result: TaskResult<R>, max_history_size: usize) -> CommResult<()> {
        if let Some(mut task) = self.processing.remove(&result.task_id) {
            task.state = if result.success {
                TaskState::Completed
            } else {
                TaskState::Failed
            };
            task.updated_at = Utc::now();

            // Add to completed history
            if self.completed.len() >= max_history_size {
                self.completed.pop_front();
            }
            self.completed.push_back(result);

            self.last_activity = Utc::now();
            Ok(())
        } else {
            bail!(CommError::ServerState {
                message: Cow::Owned(format!("Task {} not found in processing queue", result.task_id)),
            });
        }
    }

    fn get_stats(&self) -> ClientStats {
        ClientStats {
            pending_tasks: self.pending.len(),
            processing_tasks: self.processing.len(),
            completed_tasks: self.completed.len(),
            failed_tasks: self.completed.iter().filter(|r| !r.success).count(),
            last_activity: self.last_activity,
        }
    }
}

/// Statistics for a client queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientStats {
    pub pending_tasks: usize,
    pub processing_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub last_activity: chrono::DateTime<Utc>,
}

/// HTTP server for bidirectional communication.
pub struct HttpServer<T, R>
where T: Clone
{
    config: ServerConfig,
    queues: Arc<RwLock<HashMap<String, ClientQueue<T, R>>>>,
    _phantom: PhantomData<(T, R)>,
}

impl<T, R> HttpServer<T, R>
where
    T: Clone + Send + Sync + 'static + std::fmt::Debug,
    R: Clone + Send + Sync + 'static + std::fmt::Debug,
{
    /// Create a new HTTP server.
    pub fn new() -> Self {
        Self::with_config(ServerConfig::default())
    }

    /// Create a new HTTP server with custom configuration.
    pub fn with_config(config: ServerConfig) -> Self {
        info!(
            max_queue_size = config.max_queue_size,
            max_history_size = config.max_history_size,
            "HTTP server created"
        );

        Self {
            config,
            queues: Arc::new(RwLock::new(HashMap::new())),
            _phantom: PhantomData,
        }
    }

    /// Enqueue a task for a specific client.
    #[instrument(skip(self))]
    pub async fn enqueue_task_for_client(
        &self,
        client_id: &str,
        task_data: T,
    ) -> CommResult<TaskId> {
        let task = TaskItem::new(task_data);
        let task_id = task.id;

        let mut queues = self.queues.write().await;
        let client_queue = queues
            .entry(client_id.to_string())
            .or_insert_with(ClientQueue::new);

        client_queue.enqueue_task(task, self.config.max_queue_size)?;

        debug!(task_id = %task_id, client_id = %client_id, "Task enqueued for client");
        Ok(task_id)
    }

    /// Get statistics for all clients.
    pub async fn get_all_stats(&self) -> HashMap<String, ClientStats> {
        let queues = self.queues.read().await;
        queues
            .iter()
            .map(|(client_id, queue)| (client_id.clone(), queue.get_stats()))
            .collect()
    }

    /// Get statistics for a specific client.
    pub async fn get_client_stats(&self, client_id: &str) -> Option<ClientStats> {
        let queues = self.queues.read().await;
        queues.get(client_id).map(|queue| queue.get_stats())
    }

    pub async fn poll_task_internal(&self, client_id: &str) -> CommResult<Option<TaskItem<T>>> {
        let mut queues = self.queues.write().await;
        let client_queue = queues
            .entry(client_id.to_string())
            .or_insert_with(ClientQueue::new);

        let task = client_queue.dequeue_task(client_id);
        if task.is_some() {
            trace!(client_id = %client_id, "Task dequeued for client");
        }

        Ok(task)
    }

    pub async fn submit_result_internal(&self, result: TaskResult<R>) -> CommResult<()> {
        let mut queues = self.queues.write().await;

        if let Some(queue) = queues.get_mut(&result.client_id) {
            let task_id = result.task_id;
            let client_id = result.client_id.clone();
            if queue.processing.contains_key(&task_id) {
                queue.complete_task(result, self.config.max_history_size)?;
                debug!(task_id = %task_id, client_id = %client_id, "Task result processed");
                return Ok(());
            }
        }

        warn!(task_id = %result.task_id, "Task result received for unknown task");
        bail!(CommError::ServerState {
            message: Cow::Owned(format!(
                "Task result for task {} from client {} could not be processed because task or client not found",
                result.task_id, result.client_id
            )),
        });
    }
}

impl<T, R> Default for HttpServer<T, R>
where
    T: Clone + Send + Sync + 'static + std::fmt::Debug,
    R: Clone + Send + Sync + 'static + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}
