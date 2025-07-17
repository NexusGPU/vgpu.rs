//! Common types for HTTP bidirectional communication.

use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use uuid::Uuid;

/// Unique identifier for tasks.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, derive_more::Display)]
pub struct TaskId(pub Uuid);

impl TaskId {
    /// Create a new random task ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

/// State of a task in the system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    /// Task is queued and waiting to be processed
    Queued,
    /// Task is being processed by a client
    Processing,
    /// Task has been completed successfully
    Completed,
    /// Task failed during processing
    Failed,
}

/// Wrapper for task data with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskItem<T> {
    /// Unique identifier for the task
    pub id: TaskId,
    /// The actual task data
    pub data: T,
    /// Current state of the task
    pub state: TaskState,
    /// When the task was created
    pub created_at: DateTime<Utc>,
    /// When the task was last updated
    pub updated_at: DateTime<Utc>,
    /// Optional client ID that is processing this task
    pub client_id: Option<String>,
}

impl<T> TaskItem<T>
where T: Clone
{
    /// Create a new task item with the given data.
    pub fn new(data: T) -> Self {
        let now = Utc::now();
        Self {
            id: TaskId::new(),
            data,
            state: TaskState::Queued,
            created_at: now,
            updated_at: now,
            client_id: None,
        }
    }
}

/// Result of processing a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult<R> {
    /// ID of the task this result corresponds to
    pub task_id: TaskId,
    /// ID of the client that processed the task
    pub client_id: String,
    /// Whether the task was processed successfully
    pub success: bool,
    /// Optional result data
    pub result: Option<R>,
    /// Optional error message if task failed
    pub error: Option<String>,
    /// When the result was generated
    pub timestamp: DateTime<Utc>,
}

impl<R> TaskResult<R> {
    /// Create a successful task result.
    pub fn success(task_id: TaskId, client_id: impl Into<String>, result: R) -> Self {
        Self {
            task_id,
            client_id: client_id.into(),
            success: true,
            result: Some(result),
            error: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a failed task result.
    pub fn failure(
        task_id: TaskId,
        client_id: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            task_id,
            client_id: client_id.into(),
            success: false,
            result: None,
            error: Some(error.into()),
            timestamp: Utc::now(),
        }
    }
}

/// Trait for processing tasks.
pub trait TaskProcessor<T, R> {
    /// Process a task and return a result.
    ///
    /// # Arguments
    ///
    /// * `task` - The task data to process
    ///
    /// # Errors
    ///
    /// Returns an error if the task cannot be processed
    fn process_task(&self, task: &T) -> Result<R, Box<dyn std::error::Error + Send + Sync>>;
}
