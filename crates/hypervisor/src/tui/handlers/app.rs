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
use crate::tui::state::{MockWorkerMonitor, WorkerMonitor};
use crate::tui::types::RefreshEvent;

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

/// Run TUI monitor in mock mode with simulated data
pub async fn run_tui_monitor_mock() -> Result<()> {
    let mut stdout = std::io::stdout();
    enable_raw_mode()?;
    stdout.execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut monitor = MockWorkerMonitor::new();

    let (tx, mut rx) = mpsc::channel(100);

    // Start periodic refresh task for mock data updates
    let tx_clone = tx.clone();
    let _refresh_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;
            if tx_clone.send(RefreshEvent::Tick).await.is_err() {
                break;
            }
        }
    });

    // Initial mock data generation
    monitor.update_mock_workers();

    // Run the UI loop with mock data
    let result = run_mock_ui_loop(&mut terminal, &mut monitor, &mut rx).await;

    // Cleanup
    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;

    result
}

/// UI loop for mock mode
async fn run_mock_ui_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    monitor: &mut MockWorkerMonitor,
    rx: &mut mpsc::Receiver<RefreshEvent>,
) -> Result<()> {
    use crate::tui::types::AppState;
    use crossterm::event::{self, Event, KeyCode, KeyEventKind};

    loop {
        terminal.draw(|f| monitor.render(f))?;

        tokio::select! {
            event = tokio::time::timeout(Duration::from_millis(100), async { event::read() }) => {
                match event {
                    Ok(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => {
                                match monitor.app_state() {
                                    AppState::DetailDialog(_) => {
                                        monitor.close_details();
                                    }
                                    AppState::Normal => {
                                        if key.code == KeyCode::Char('q') {
                                            return Ok(());
                                        }
                                    }
                                }
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                if matches!(monitor.app_state(), AppState::Normal) {
                                    monitor.next();
                                }
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                if matches!(monitor.app_state(), AppState::Normal) {
                                    monitor.previous();
                                }
                            }
                            KeyCode::Enter => {
                                if matches!(monitor.app_state(), AppState::Normal) {
                                    monitor.show_details();
                                }
                            }
                            KeyCode::Char('r') => {
                                if matches!(monitor.app_state(), AppState::Normal) {
                                    monitor.update_mock_workers();
                                }
                            }
                            _ => {}
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::error!("Terminal event error: {}", e);
                    }
                    Err(_) => {} // Timeout
                    _ => {}
                }
            }
            refresh_event = rx.recv() => {
                match refresh_event {
                    Some(RefreshEvent::FileSystemChange) | Some(RefreshEvent::Tick) => {
                        monitor.update_mock_workers();
                    }
                    None => return Ok(()),
                }
            }
        }
    }
}
