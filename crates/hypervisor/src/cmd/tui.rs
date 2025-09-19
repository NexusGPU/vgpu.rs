use crate::config::ShowTuiWorkersArgs;
use crate::tui;
use anyhow::Result;

pub async fn run_show_tui_workers(args: ShowTuiWorkersArgs) -> Result<()> {
    utils::logging::init_with_log_path(args.log_path);

    tracing::info!(
        "Starting shared memory TUI monitor with pattern: {}",
        args.glob
    );

    tui::run_shm_tui_monitor(args.glob, args.refresh_interval).await
}
