//! command handler using the http-bidir-comm library.
//!
//! This module demonstrates how to replace the existing command_handler implementation
//! with the new generic HTTP bidirectional communication library.

use std::ffi::c_int;
use std::sync::Arc;

use api_types::LimiterCommand;
use api_types::LimiterCommandResponse;
use api_types::LimiterCommandType;
use error_stack::Report;
use http_bidir_comm::BlockingHttpClient;
use http_bidir_comm::ClientConfig;
use http_bidir_comm::TaskProcessor;
use tracing::error;
use tracing::info;
use tracing::instrument;

use crate::GLOBAL_NGPU_LIBRARY;

/// Mock implementations of the C functions for testing purposes.
///
/// This module is only compiled when running tests (`#[cfg(test)]`).
/// It provides mock versions of the FFI functions, allowing us to control their
/// return values and test different scenarios without relying on the actual
/// host process. We use `#[no_mangle]` to ensure the function names are not
/// changed by the Rust compiler, so the linker can find them.
#[cfg(test)]
mod ffi_test_helpers {
    use std::os::raw::c_int;
    use std::sync::atomic::AtomicI32;
    use std::sync::atomic::Ordering;

    // Atomic variables to control the mock return values from within tests.
    // This allows us to simulate both success (0) and failure (non-zero) cases.
    pub static MOCK_TF_HEALTH_CHECK_RESULT: AtomicI32 = AtomicI32::new(0);
    pub static MOCK_TF_SUSPEND_RESULT: AtomicI32 = AtomicI32::new(0);
    pub static MOCK_TF_RESUME_RESULT: AtomicI32 = AtomicI32::new(0);
    pub static MOCK_TF_VRAM_RECLAIM_RESULT: AtomicI32 = AtomicI32::new(0);

    #[no_mangle]
    pub unsafe extern "C" fn tf_health_check() -> c_int {
        MOCK_TF_HEALTH_CHECK_RESULT.load(Ordering::SeqCst)
    }

    #[no_mangle]
    pub unsafe extern "C" fn tf_suspend() -> c_int {
        MOCK_TF_SUSPEND_RESULT.load(Ordering::SeqCst)
    }

    #[no_mangle]
    pub unsafe extern "C" fn tf_resume() -> c_int {
        MOCK_TF_RESUME_RESULT.load(Ordering::SeqCst)
    }

    #[no_mangle]
    pub unsafe extern "C" fn tf_vram_reclaim() -> c_int {
        MOCK_TF_VRAM_RECLAIM_RESULT.load(Ordering::SeqCst)
    }
}

/// command processor using the new HTTP bidirectional communication library.
#[derive(Clone)]
pub struct CommandProcessor;

impl TaskProcessor<LimiterCommand, LimiterCommandResponse> for CommandProcessor {
    fn process_task(
        &self,
        task: &LimiterCommand,
    ) -> Result<LimiterCommandResponse, Box<dyn std::error::Error + Send + Sync>> {
        info!(command_id = task.id, command_type = ?task.kind, "Processing command");

        let (ret, desc) = {
            unsafe {
                type TfHealthCheck = unsafe extern "C" fn() -> c_int;
                type TfSuspend = unsafe extern "C" fn() -> c_int;
                type TfResume = unsafe extern "C" fn() -> c_int;
                type TfVramReclaim = unsafe extern "C" fn() -> c_int;

                match task.kind {
                    LimiterCommandType::TfHealthCheck => {
                        let lib = GLOBAL_NGPU_LIBRARY.get().unwrap();
                        let symbol: libloading::Symbol<TfHealthCheck> =
                            lib.get(b"tf_health_check").unwrap();
                        let result = symbol();
                        (result, "tf_health_check")
                    }
                    LimiterCommandType::TfSuspend => {
                        let lib = GLOBAL_NGPU_LIBRARY.get().unwrap();
                        let symbol: libloading::Symbol<TfSuspend> = lib.get(b"tf_suspend").unwrap();
                        let result = symbol();
                        (result, "tf_suspend")
                    }
                    LimiterCommandType::TfResume => {
                        let lib = GLOBAL_NGPU_LIBRARY.get().unwrap();
                        let symbol: libloading::Symbol<TfResume> = lib.get(b"tf_resume").unwrap();
                        let result = symbol();
                        (result, "tf_resume")
                    }
                    LimiterCommandType::TfVramReclaim => {
                        let lib = GLOBAL_NGPU_LIBRARY.get().unwrap();
                        let symbol: libloading::Symbol<TfVramReclaim> =
                            lib.get(b"tf_vram_reclaim").unwrap();
                        let result = symbol();
                        (result, "tf_vram_reclaim")
                    }
                }
            }
        };

        let success = ret == 0;
        if success {
            info!(
                command = desc,
                command_id = task.id,
                "Command executed successfully"
            );
            Ok(LimiterCommandResponse {
                id: task.id,
                success: true,
                message: None,
            })
        } else {
            error!(
                command = desc,
                code = ret,
                command_id = task.id,
                "Command execution failed"
            );
            Ok(LimiterCommandResponse {
                id: task.id,
                success: false,
                message: Some(format!("{desc} failed with code {ret}")),
            })
        }
    }
}

/// command handler configuration.
#[derive(Debug, Clone)]
pub struct CommandHandlerConfig {
    /// Unique limiter ID
    pub limiter_id: String,
    client_config: ClientConfig,
}

impl CommandHandlerConfig {
    /// Create a new configuration with sensible defaults.
    pub fn new(host_pid: u32, client_config: ClientConfig) -> Self {
        Self {
            client_config,
            limiter_id: format!("limiter_{host_pid}"),
        }
    }
}

/// command handler that uses the HTTP bidirectional communication library.
pub struct CommandHandler {
    client: BlockingHttpClient<LimiterCommand, LimiterCommandResponse>,
    processor: Arc<CommandProcessor>,
}

impl CommandHandler {
    /// Create a new command handler.
    ///
    /// # Arguments
    ///
    /// * `config` - Command handler configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created
    pub fn new(config: CommandHandlerConfig) -> Result<Self, Report<http_bidir_comm::CommError>> {
        info!(
            server_url = %config.client_config.server_url,
            limiter_id = %config.limiter_id,
            "Creating command handler"
        );

        let client = BlockingHttpClient::new(config.client_config)?;
        let processor = Arc::new(CommandProcessor);

        Ok(Self { client, processor })
    }

    /// Start the command handler.
    ///
    /// This method runs indefinitely, polling for commands and processing them.
    /// It includes automatic reconnection and error recovery.
    ///
    /// # Errors
    ///
    /// Returns an error if the handler encounters a fatal error that prevents it from continuing
    #[instrument(skip(self))]
    pub fn start(&self) -> Result<(), Report<http_bidir_comm::CommError>> {
        info!("Starting command handler");

        // Use the same API path as the dispatcher
        self.client
            .start("/api/v1/limiter", Arc::clone(&self.processor))
    }
}

/// Start the background command handler with default configuration.
///
/// This is a drop-in replacement for the original `start_background_handler` function.
///
/// # Arguments
///
/// * `ip` - Hypervisor IP address
/// * `port` - Hypervisor port
pub fn start_background_handler(ip: &str, port: &str, host_pid: u32) {
    let config =
        CommandHandlerConfig::new(host_pid, ClientConfig::new(format!("http://{ip}:{port}")));

    std::thread::spawn(move || match CommandHandler::new(config) {
        Ok(handler) => {
            if let Err(e) = handler.start() {
                error!(error = %e, "command handler failed");
            }
        }
        Err(e) => {
            error!(error = %e, "Failed to create command handler");
        }
    });
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn process_command() {
        use std::sync::atomic::Ordering;

        use super::ffi_test_helpers::*;

        let processor = CommandProcessor;

        let command = LimiterCommand {
            id: 1,
            kind: LimiterCommandType::TfHealthCheck,
        };

        // --- Test Success Case ---
        // Set the mock function to return 0 (success)
        MOCK_TF_HEALTH_CHECK_RESULT.store(0, Ordering::SeqCst);

        let result = processor.process_task(&command);
        assert!(result.is_ok(), "process_task should not error out");

        let response = result.unwrap();
        assert_eq!(response.id, 1);
        assert!(response.success, "Command should be successful");
        assert!(
            response.message.is_none(),
            "Success response should not have a message"
        );

        // --- Test Failure Case ---
        // Set the mock function to return a non-zero error code
        MOCK_TF_HEALTH_CHECK_RESULT.store(-1, Ordering::SeqCst);

        let result_fail = processor.process_task(&command);
        assert!(result_fail.is_ok());

        let response_fail = result_fail.unwrap();
        assert_eq!(response_fail.id, 1);
        assert!(!response_fail.success, "Command should fail");
        assert!(
            response_fail.message.is_some(),
            "Failure response should have a message"
        );
        assert_eq!(
            response_fail.message.unwrap(),
            "tf_health_check failed with code -1"
        );
    }
}
