use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::path::Path;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::ui::tui::state::WorkerMonitor;
use crate::ui::tui::types::{AppState, RefreshEvent};

pub async fn setup_file_watcher(pattern: &str, tx: mpsc::Sender<RefreshEvent>) -> Result<()> {
    let path = if pattern.starts_with("/dev/shm/") {
        "/dev/shm"
    } else {
        pattern.split('/').next().unwrap_or("/dev/shm")
    };

    let (watch_tx, mut watch_rx) = mpsc::channel(100);
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(event) = res {
                if let Err(e) = watch_tx.blocking_send(event) {
                    tracing::error!("Failed to send event to watch_tx, error: {e:?}");
                }
            }
        },
        notify::Config::default(),
    )?;

    watcher.watch(Path::new(path), RecursiveMode::NonRecursive)?;

    while let Some(_event) = watch_rx.recv().await {
        if tx.send(RefreshEvent::FileSystemChange).await.is_err() {
            break;
        }
    }

    Ok(())
}

pub async fn run_ui_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    monitor: &mut WorkerMonitor,
    rx: &mut mpsc::Receiver<RefreshEvent>,
    glob_pattern: &str,
) -> Result<()> {
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
                                    if let Err(e) = monitor.update_workers(glob_pattern) {
                                        tracing::error!("Failed to update workers: {}", e);
                                    }
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
                        if let Err(e) = monitor.update_workers(glob_pattern) {
                            tracing::error!("Failed to update workers: {}", e);
                        }
                    }
                    None => return Ok(()),
                }
            }
        }
    }
}
