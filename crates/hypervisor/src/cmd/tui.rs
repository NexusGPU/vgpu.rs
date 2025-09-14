use crate::config::ShowTuiWorkersArgs;
use crate::tui;
use anyhow::Result;

pub async fn run_show_tui_workers(args: ShowTuiWorkersArgs) -> Result<()> {
    utils::logging::init_with_log_path(args.log_path);

    tracing::info!("Starting TUI worker monitor with pattern: {}", args.glob);

    if args.mock {
        tracing::info!("Starting TUI in mock mode for local debugging");
        tui::handlers::run_tui_monitor_mock().await
    } else {
        tui::handlers::run_tui_monitor(format!("/dev/shm/{}", args.glob)).await
    }
}
