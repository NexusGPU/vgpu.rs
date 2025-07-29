use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
pub struct ShowShmArgs {
    #[arg(long, help = "Shared memory identifier")]
    pub shm_identifier: String,
}

#[derive(Parser)]
pub struct ShowTuiWorkersArgs {
    #[arg(
        long,
        help = "Glob pattern for worker shared memory files",
        default_value = "/tf_shm_*"
    )]
    pub glob: String,

    #[arg(
        long,
        help = "Log path for TUI worker monitor",
        default_value = "/tmp/hypervisor_tui_workers.log"
    )]
    pub log_path: String,

    #[arg(
        long,
        help = "Enable mock mode for TUI development and testing",
        default_value_t = false,
        action = clap::ArgAction::SetTrue
    )]
    pub mock: bool,
}

#[derive(Parser)]
pub struct MountShmArgs {
    #[arg(
        long,
        help = "Shared memory mount point path",
        default_value = "/run/tensor-fusion/shm"
    )]
    pub mount_point: PathBuf,

    #[arg(long, help = "Shared memory size in MB", default_value = "64")]
    pub size_mb: u64,
}
