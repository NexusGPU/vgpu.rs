use clap::{Parser, Subcommand};
use utils::version;

use crate::config::daemon::DaemonArgs;
use crate::config::shm::{MountShmArgs, ShowShmArgs, ShowTuiWorkersArgs};

#[derive(Parser)]
#[command(about, long_about, version = &**version::VERSION)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run hypervisor daemon
    Daemon(Box<DaemonArgs>),
    /// Mount shared memory to host path
    #[command(name = "mount-shm")]
    MountShm(MountShmArgs),
    /// Show shared memory state
    #[command(name = "show-shm")]
    ShowShm(ShowShmArgs),
    /// Show TUI for monitoring workers
    #[command(name = "tui")]
    ShowTuiWorkers(ShowTuiWorkersArgs),
}
