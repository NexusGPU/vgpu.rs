mod api;
mod app;
mod app_builder;
mod cli;
mod config;
mod core;
mod domain;
mod infrastructure;
mod ui;

// Re-export main modules for backward compatibility
pub use domain::hypervisor;
pub use domain::pod_management;
pub use domain::process;
pub use domain::scheduler;
pub use infrastructure::gpu_init;
pub use infrastructure::gpu_observer;
pub use infrastructure::gpu_watcher;
pub use infrastructure::host_pid_probe;
pub use infrastructure::k8s;
pub use infrastructure::kube_client;
pub use infrastructure::limiter_comm;
pub use infrastructure::logging;
pub use infrastructure::metrics;

use anyhow::Result;
use clap::Parser;

use crate::cli::commands::{DaemonCommand, MountShmCommand, ShowShmCommand, ShowTuiWorkersCommand};
use crate::config::{Cli, Commands};
use crate::core::command::Command;

/// Sets up global panic hooks.
fn setup_global_hooks() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        default_hook(panic_info);
        tracing::error!("Thread panicked: {}", panic_info);
    }));
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_global_hooks();

    let cli = Cli::parse();

    // Execute the appropriate command based on CLI input
    match cli.command {
        Commands::Daemon(daemon_args) => {
            let command = DaemonCommand::new(*daemon_args);
            command.execute().await
        }
        Commands::MountShm(mount_shm_args) => {
            let command = MountShmCommand::new(mount_shm_args);
            command.execute().await
        }
        Commands::ShowShm(show_shm_args) => {
            let command = ShowShmCommand::new(show_shm_args);
            command.execute().await
        }
        Commands::ShowTuiWorkers(show_tui_workers_args) => {
            let command = ShowTuiWorkersCommand::new(show_tui_workers_args);
            command.execute().await
        }
    }
}
