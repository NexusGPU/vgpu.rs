use derive_more::Display;

/// Error type shared by limiter and hypervisor components.
#[derive(Debug, Display)]
pub enum ErlError {
    /// Underlying shared storage failed.
    #[display("storage access failed: {reason}")]
    StorageFailure { reason: String },
    /// Configuration is invalid or inconsistent.
    #[display("invalid configuration: {reason}")]
    InvalidConfiguration { reason: String },
}

impl core::error::Error for ErlError {}

impl ErlError {
    pub fn storage(reason: impl Into<String>) -> Self {
        Self::StorageFailure {
            reason: reason.into(),
        }
    }

    pub fn invalid_config(reason: impl Into<String>) -> Self {
        Self::InvalidConfiguration {
            reason: reason.into(),
        }
    }
}
