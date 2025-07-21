//! Synchronous Event-Source client implementation.
//!
//! This client uses **blocking** API and depends on `eventsource::reqwest::Client` (internal
//! uses `reqwest::blocking`). It will automatically reconnect with exponential backoff when
//! the connection is lost.

use core::fmt::Debug;
use core::marker::PhantomData;
use std::sync::Arc;
use std::thread::sleep;

use error_stack::ResultExt;
use eventsource::reqwest::Client as EsClient;
use reqwest::blocking::Client as BlockingClient;
use tracing::debug;
use tracing::info;
use tracing::trace;
use tracing::warn;
use url::Url;

use crate::error::CommError;
use crate::error::CommResult;
use crate::types::TaskItem;
use crate::types::TaskProcessor;
use crate::types::TaskResult;
use crate::ClientConfig;

/// blocking SSE client
pub struct BlockingSseClient<T, R> {
    config: ClientConfig,
    /// only used to submit results
    http: BlockingClient,
    _phantom: PhantomData<(T, R)>,
}

impl<T, R> BlockingSseClient<T, R>
where
    T: for<'de> serde::Deserialize<'de> + serde::Serialize + Debug + Clone + 'static,
    R: serde::Serialize + for<'de> serde::Deserialize<'de> + Debug + Clone + 'static,
{
    /// create client
    pub fn new(config: ClientConfig) -> CommResult<Self> {
        let http = BlockingClient::builder()
            .timeout(config.request_timeout)
            .build()
            .change_context(CommError::Configuration {
                message: "Failed to create blocking HTTP client for SSE".into(),
            })?;

        info!(server_url = %config.server_url, client_id = %config.client_id, "Blocking SSE client created");

        Ok(Self {
            config,
            http,
            _phantom: PhantomData,
        })
    }

    /// start event loop (block current thread)
    pub fn start<P>(&self, base_path: impl AsRef<str>, processor: Arc<P>) -> CommResult<()>
    where
        P: TaskProcessor<T, R> + Send + Sync + 'static,
    {
        let base_path = base_path.as_ref();
        let events_url = format!(
            "{}{}/events/{}",
            self.config.server_url, base_path, self.config.client_id
        );
        let result_url = format!("{}{}/result", self.config.server_url, base_path);

        let mut delay = self.config.retry_delay;

        loop {
            info!(url = %events_url, "Connecting to SSE endpoint (blocking)");
            let url = Url::parse(&events_url).change_context(CommError::Configuration {
                message: "Invalid events URL".into(),
            })?;
            let mut stream = EsClient::new(url);

            // iterate events
            for event in &mut stream {
                match event {
                    Ok(msg) => {
                        trace!(data = %msg.data, "Received SSE message");
                        if msg.data.trim().is_empty() {
                            continue;
                        }

                        let task: TaskItem<T> = serde_json::from_str(&msg.data).change_context(
                            CommError::Serialization {
                                message: "Failed to deserialize TaskItem".into(),
                            },
                        )?;

                        debug!(task_id = %task.id, "Processing task");
                        let result = match processor.process_task(&task.data) {
                            Ok(r) => TaskResult::success(task.id, &self.config.client_id, r),
                            Err(e) => {
                                TaskResult::failure(task.id, &self.config.client_id, e.to_string())
                            }
                        };

                        // send result
                        self.http
                            .post(&result_url)
                            .json(&result)
                            .send()
                            .change_context(CommError::Network {
                                message: "Failed to send result".into(),
                            })?;
                    }
                    Err(e) => {
                        warn!("SSE stream error: {e}");
                        break; // exit for loop to reconnect
                    }
                }
            }

            // fall-through means we exited the loop, will sleep and reconnect

            warn!(
                delay_sec = delay.as_secs_f32(),
                "SSE disconnected, retrying after delay"
            );
            sleep(delay);
            delay = std::cmp::min(delay * 2, self.config.max_retry_delay);
        }
    }
}
