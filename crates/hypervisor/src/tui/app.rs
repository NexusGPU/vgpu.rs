use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::widgets::TableState;
use ratatui::Terminal;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::tui::dialog::ShmDetailDialog;
use crate::tui::reader::ShmReader;
use crate::tui::table::ShmTable;
use crate::tui::types::{AppState, RefreshEvent, ShmEntry};

pub struct ShmMonitorApp {
    entries: Vec<ShmEntry>,
    table_state: TableState,
    selected_index: usize,
    app_state: AppState,
    pattern: String,
}

impl ShmMonitorApp {
    pub fn new(pattern: String) -> Self {
        Self {
            entries: Vec::new(),
            table_state: TableState::default(),
            selected_index: 0,
            app_state: AppState::Normal,
            pattern,
        }
    }

    pub fn update_entries(&mut self) -> Result<()> {
        match ShmReader::read_all_shm_entries(&self.pattern) {
            Ok(entries) => {
                self.entries = entries;
                self.update_selection();

                // If we're currently showing a detail dialog, update it with fresh data
                self.refresh_detail_dialog();

                Ok(())
            }
            Err(e) => {
                tracing::error!("Failed to read shared memory entries: {}", e);
                Err(e)
            }
        }
    }

    pub fn next(&mut self) {
        let entry_count = self.entries.len();
        if entry_count > 0 {
            self.selected_index = (self.selected_index + 1) % entry_count;
            self.table_state.select(Some(self.selected_index));
        }
    }

    pub fn previous(&mut self) {
        let entry_count = self.entries.len();
        if entry_count > 0 {
            if self.selected_index == 0 {
                self.selected_index = entry_count - 1;
            } else {
                self.selected_index -= 1;
            }
            self.table_state.select(Some(self.selected_index));
        }
    }

    pub fn show_details(&mut self) {
        if let Some(entry) = self.entries.get(self.selected_index).cloned() {
            self.app_state = AppState::DetailDialog(entry);
        }
    }

    pub fn close_details(&mut self) {
        self.app_state = AppState::Normal;
    }

    fn update_selection(&mut self) {
        let entry_count = self.entries.len();
        if entry_count == 0 {
            self.selected_index = 0;
            self.table_state.select(None);
        } else {
            if self.selected_index >= entry_count {
                self.selected_index = entry_count - 1;
            }
            self.table_state.select(Some(self.selected_index));
        }
    }

    fn refresh_detail_dialog(&mut self) {
        if let AppState::DetailDialog(ref current_entry) = &self.app_state {
            // Find the updated entry that matches the current dialog entry
            if let Some(updated_entry) = self
                .entries
                .iter()
                .find(|entry| entry.pod_identifier == current_entry.pod_identifier)
                .cloned()
            {
                // Update the dialog with fresh data
                self.app_state = AppState::DetailDialog(updated_entry);
            } else {
                // If the entry no longer exists, close the dialog
                self.app_state = AppState::Normal;
            }
        }
    }

    pub fn render(&mut self, frame: &mut ratatui::Frame) {
        let area = frame.area();

        ShmTable::render(
            &self.entries,
            &mut self.table_state,
            &self.app_state,
            frame,
            area,
        );

        if let AppState::DetailDialog(ref entry) = self.app_state {
            ShmDetailDialog::render(entry, frame, area);
        }
    }
}

pub async fn run_shm_tui_monitor(pattern: String, refresh_interval_secs: u64) -> Result<()> {
    let mut stdout = std::io::stdout();
    enable_raw_mode()?;
    stdout.execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = ShmMonitorApp::new(pattern.clone());

    let (tx, mut rx) = mpsc::channel(100);

    // Start periodic refresh task with configurable interval
    let _refresh_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(refresh_interval_secs));
        loop {
            interval.tick().await;
            tracing::debug!("Sending refresh tick");
            if tx.send(RefreshEvent::Tick).await.is_err() {
                tracing::debug!("Failed to send refresh tick, receiver dropped");
                break;
            }
        }
    });

    if let Err(e) = app.update_entries() {
        tracing::error!("Initial entry update failed: {}", e);
    }

    let result = run_event_loop(&mut terminal, &mut app, &mut rx).await;

    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;

    result
}

async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut ShmMonitorApp,
    rx: &mut mpsc::Receiver<RefreshEvent>,
) -> Result<()> {
    loop {
        terminal.draw(|f| app.render(f))?;

        // Handle keyboard input
        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => match &app.app_state {
                        AppState::DetailDialog(_) => {
                            app.close_details();
                        }
                        AppState::Normal => {
                            if key.code == KeyCode::Char('q') {
                                return Ok(());
                            }
                        }
                    },
                    KeyCode::Down | KeyCode::Char('j') => {
                        if matches!(app.app_state, AppState::Normal) {
                            app.next();
                        }
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        if matches!(app.app_state, AppState::Normal) {
                            app.previous();
                        }
                    }
                    KeyCode::Enter => {
                        if matches!(app.app_state, AppState::Normal) {
                            app.show_details();
                        }
                    }
                    KeyCode::Char('r') => {
                        if matches!(app.app_state, AppState::Normal) {
                            if let Err(e) = app.update_entries() {
                                tracing::error!("Failed to update entries: {}", e);
                            }
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        } else {
            // Handle refresh events
            match rx.try_recv() {
                Ok(RefreshEvent::Tick) => {
                    tracing::debug!("Received refresh tick, updating entries");
                    if let Err(e) = app.update_entries() {
                        tracing::error!("Failed to update entries: {}", e);
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    // No refresh event available, continue
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    tracing::debug!("Refresh channel disconnected");
                    return Ok(());
                }
            }
        }
    }
}
