use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
pub struct ShowShmArgs {
    #[arg(long, help = "Pod namespace")]
    pub namespace: String,

    #[arg(long, help = "Pod name")]
    pub pod_name: String,

    #[arg(
        long,
        help = "Base path for shared memory files",
        env = "SHM_BASE_PATH",
        value_hint = clap::ValueHint::DirPath,
        default_value = "/run/tensor-fusion/shm"
    )]
    pub shm_base_path: PathBuf,
}

#[derive(Parser)]
pub struct ShowTuiWorkersArgs {
    #[arg(
        long,
        help = "Glob pattern for worker shared memory files",
        default_value = "/run/tensor-fusion/shm/*/*"
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
        help = "Refresh interval in seconds for TUI data updates",
        default_value = "1"
    )]
    pub refresh_interval: u64,
}

#[derive(Parser)]
pub struct MountShmArgs {
    #[arg(
        long,
        help = "Shared memory mount point path",
        default_value = "/run/tensor-fusion/shm"
    )]
    pub mount_point: PathBuf,

    #[arg(long, help = "Shared memory size in MB", default_value = "128")]
    pub size_mb: u64,
}
