use anyhow::Result;

use crate::app::services::ApplicationServices;
use crate::app::tasks::Tasks;
use crate::config::DaemonArgs;

/// Application core structure with explicit dependencies
pub struct Application {
    services: ApplicationServices,
    daemon_args: DaemonArgs,
}

impl Application {
    /// Create new application with explicit service dependencies
    pub fn new(services: ApplicationServices, daemon_args: DaemonArgs) -> Self {
        Self {
            services,
            daemon_args,
        }
    }

    /// Get access to services
    pub fn services(&self) -> &ApplicationServices {
        &self.services
    }

    /// Get daemon arguments
    pub fn daemon_args(&self) -> &DaemonArgs {
        &self.daemon_args
    }

    /// Run application, start all tasks and wait for completion
    pub async fn run(&self) -> Result<()> {
        tracing::info!("Starting all application tasks...");

        // Create task manager
        let mut tasks = Tasks::new();

        // Start all background tasks
        if let Err(e) = tasks.spawn_all_tasks(self) {
            tracing::error!("Failed to spawn application tasks: {}", e);
            return Err(e);
        }

        // Wait for tasks to complete or receive shutdown signal
        if let Err(e) = tasks.wait_for_completion().await {
            tracing::error!("Error during task execution: {}", e);
            return Err(e);
        }

        tracing::info!("Application run completed");
        Ok(())
    }

    /// Gracefully shutdown application
    pub async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down application...");

        // Shutdown host PID probe
        self.services.host_pid_probe.shutdown().await;

        tracing::info!("Application shutdown completed");
        Ok(())
    }
}
