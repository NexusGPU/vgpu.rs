//! Error types for HTTP bidirectional communication.

use core::error::Error;

use derive_more::Display;
use error_stack::Report;

/// Result type for communication operations.
pub type CommResult<T> = Result<T, Report<CommError>>;

/// Errors that can occur during HTTP bidirectional communication.
#[derive(Debug, Display)]
pub enum CommError {
    /// Network connectivity issues
    #[display("Network error: {message}")]
    Network { message: String },

    /// HTTP request/response errors
    #[display("HTTP error: {status} - {message}")]
    Http { status: u16, message: String },

    /// Serialization/deserialization errors
    #[display("Serialization error: {message}")]
    Serialization { message: String },

    /// Configuration errors
    #[display("Configuration error: {message}")]
    Configuration { message: String },

    /// Timeout errors
    #[display("Operation timed out after {seconds}s")]
    Timeout { seconds: u64 },

    /// Client not connected
    #[display("Client is not connected")]
    NotConnected,

    /// Task processing errors
    #[display("Task processing error: {message}")]
    TaskProcessing { message: String },

    /// Server state errors
    #[display("Server state error: {message}")]
    ServerState { message: String },
}

impl Error for CommError {}
