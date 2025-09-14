use anyhow::Result;
use clap::Parser;

use hypervisor::cmd;
use hypervisor::config::{Cli, Commands};

/// Sets up global panic hooks.
fn setup_panic() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        crossterm::terminal::disable_raw_mode().unwrap();
        crossterm::execute!(std::io::stderr(), crossterm::terminal::LeaveAlternateScreen).unwrap();
        original_hook(panic_info);
    }));
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_panic();

    let cli = Cli::parse();

    match cli.command {
        Commands::Daemon(daemon_args) => cmd::run_daemon(*daemon_args).await,
        Commands::MountShm(mount_shm_args) => cmd::run_mount_shm(mount_shm_args).await,
        Commands::ShowShm(show_shm_args) => cmd::run_show_shm(show_shm_args).await,
        Commands::ShowTuiWorkers(show_tui_workers_args) => {
            cmd::run_show_tui_workers(show_tui_workers_args).await
        }
    }
}
