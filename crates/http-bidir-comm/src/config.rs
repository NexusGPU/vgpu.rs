//! client config
//!
//! This config is shared by blocking SSE client and other implementations.

use std::time::Duration;

/// HTTP/SSE client config.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// server base url
    pub server_url: String,
    /// client id
    pub client_id: String,
    /// HTTP request timeout
    pub request_timeout: Duration,
    /// poll/reconnect interval
    pub poll_interval: Duration,
    /// max retries
    pub max_retries: u32,
    /// initial retry delay
    pub retry_delay: Duration,
    /// max retry delay
    pub max_retry_delay: Duration,
}

impl ClientConfig {
    /// create new client config with default parameters.
    pub fn new(server_url: impl Into<String>) -> Self {
        Self {
            server_url: server_url.into(),
            // TODO: FIX THIS ID
            client_id: format!("client_{}", std::process::id()),
            request_timeout: Duration::from_secs(30),
            poll_interval: Duration::from_secs(1),
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
            max_retry_delay: Duration::from_secs(30),
        }
    }

    /// set client id.
    pub fn with_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = client_id.into();
        self
    }

    /// set request timeout.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// set poll/reconnect interval.
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// set retry_config
    pub fn with_retry_config(
        mut self,
        max_retries: u32,
        base_delay: Duration,
        max_delay: Duration,
    ) -> Self {
        self.max_retries = max_retries;
        self.retry_delay = base_delay;
        self.max_retry_delay = max_delay;
        self
    }
}
