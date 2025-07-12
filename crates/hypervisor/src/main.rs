mod api;
mod app;
mod app_builder;
mod config;
mod gpu_init;
mod gpu_observer;
mod host_pid_probe;
mod hypervisor;
mod k8s;
mod limiter_comm;
mod limiter_coordinator;
mod logging;
mod metrics;
mod process;
mod scheduler;
mod worker_manager;
mod worker_registration;

use anyhow::Result;
use clap::Parser;
use utils::version;

use crate::app_builder::ApplicationBuilder;
use crate::config::Cli;

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
    let _guard = logging::init(cli.gpu_metrics_file.clone());

    tracing::info!("Starting hypervisor {}", &**version::VERSION);

    let app = ApplicationBuilder::new(cli).build().await?;

    app.run().await?;
    app.shutdown().await?;

    Ok(())
}
