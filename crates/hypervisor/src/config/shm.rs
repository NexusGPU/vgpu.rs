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
        default_value = "/dev/shm/tf_shm_*"
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

    #[arg(
        long,
        help = "Cleanup all shared memory files with this base path (e.g., 'tf_shm', 'my_app')",
        value_hint = clap::ValueHint::Other,
    )]
    pub cleanup_prefix: Option<String>,
}
