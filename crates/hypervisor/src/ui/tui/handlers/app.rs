use anyhow::Result;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::time::Duration;
use tokio::sync::mpsc;

use super::event_loop::{run_ui_loop, setup_file_watcher};
use crate::ui::tui::state::WorkerMonitor;
use crate::ui::tui::types::RefreshEvent;

pub async fn run_tui_monitor(glob_pattern: String) -> Result<()> {
    let mut stdout = std::io::stdout();
    enable_raw_mode()?;
    stdout.execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut monitor = WorkerMonitor::new();

    let (tx, mut rx) = mpsc::channel(100);

    // Start file watcher task
    let pattern_clone = glob_pattern.clone();
    let tx_clone = tx.clone();
    let _watcher_task = tokio::spawn(async move {
        if let Err(e) = setup_file_watcher(&pattern_clone, tx_clone).await {
            tracing::error!("File watcher error: {}", e);
        }
    });

    // Start periodic refresh task
    let tx_clone = tx.clone();
    let _refresh_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            if tx_clone.send(RefreshEvent::Tick).await.is_err() {
                break;
            }
        }
    });

    // Initial worker update
    monitor.update_workers(&glob_pattern)?;

    // Run the UI loop
    let result = run_ui_loop(&mut terminal, &mut monitor, &mut rx, &glob_pattern).await;

    // Cleanup
    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;

    result
}
