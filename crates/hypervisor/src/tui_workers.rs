use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use glob::glob;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Row, Table, TableState};
use tokio::sync::mpsc;
use utils::shared_memory::handle::SharedMemoryHandle;

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub uuid: String,
    pub available_cuda_cores: i32,
    pub total_cuda_cores: u32,
    pub mem_limit: u64,
    pub pod_memory_used: u64,
    pub up_limit: u32,
}

#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub identifier: String,
    pub devices: Vec<DeviceInfo>,
    pub is_healthy: bool,
}

pub struct WorkerMonitor {
    workers: HashMap<String, WorkerInfo>,
    table_state: TableState,
    selected_index: usize,
}

impl WorkerMonitor {
    pub fn new() -> Self {
        Self {
            workers: HashMap::new(),
            table_state: TableState::default(),
            selected_index: 0,
        }
    }

    pub fn update_workers(&mut self, pattern: &str) -> Result<()> {
        let mut new_workers = HashMap::new();

        for entry in glob(pattern).context("Failed to parse glob pattern")? {
            let path = entry.context("Failed to read glob entry")?;

            if !path.is_file() {
                continue;
            }

            let identifier = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("")
                .to_string();

            if identifier.is_empty() {
                continue;
            }

            match self.read_worker_info(&identifier, &path) {
                Ok(info) => {
                    new_workers.insert(identifier.clone(), info);
                }
                Err(e) => {
                    tracing::warn!("Failed to read worker {}: {}", identifier, e);
                    continue;
                }
            }
        }

        self.workers = new_workers;
        self.update_selection();
        Ok(())
    }

    fn read_worker_info(&self, identifier: &str, _path: &Path) -> Result<WorkerInfo> {
        let handle =
            SharedMemoryHandle::open(identifier).context("Failed to open shared memory")?;

        let state = handle.get_state();
        let is_healthy = state.is_healthy(60); // 60 seconds timeout

        // Read all active devices
        let mut devices = Vec::new();
        let device_count = state.device_count();

        for i in 0..device_count {
            let device = &state.devices[i];
            if device.is_active() {
                let device_info = &device.device_info;
                devices.push(DeviceInfo {
                    uuid: device.get_uuid_owned(),
                    available_cuda_cores: device_info.get_available_cores(),
                    total_cuda_cores: device_info.get_total_cores(),
                    mem_limit: device_info.get_mem_limit(),
                    pod_memory_used: device_info.get_pod_memory_used(),
                    up_limit: device_info.get_up_limit(),
                });
            }
        }

        Ok(WorkerInfo {
            identifier: identifier.to_string(),
            devices,
            is_healthy,
        })
    }

    fn update_selection(&mut self) {
        let worker_count = self.workers.len();
        if worker_count == 0 {
            self.selected_index = 0;
            self.table_state.select(None);
        } else {
            if self.selected_index >= worker_count {
                self.selected_index = worker_count - 1;
            }
            self.table_state.select(Some(self.selected_index));
        }
    }

    pub fn next(&mut self) {
        let worker_count = self.workers.len();
        if worker_count > 0 {
            self.selected_index = (self.selected_index + 1) % worker_count;
            self.table_state.select(Some(self.selected_index));
        }
    }

    pub fn previous(&mut self) {
        let worker_count = self.workers.len();
        if worker_count > 0 {
            if self.selected_index == 0 {
                self.selected_index = worker_count - 1;
            } else {
                self.selected_index -= 1;
            }
            self.table_state.select(Some(self.selected_index));
        }
    }

    pub fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();

        let header_cells = [
            "Worker",
            "Device UUID",
            "Available/Total Cores",
            "Memory Used/Limit",
            "Up Limit",
            "Health",
        ]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow)));

        let header = Row::new(header_cells).height(1).bottom_margin(1);

        // Collect all devices from all workers
        let mut device_rows = Vec::new();
        let mut workers: Vec<_> = self.workers.values().collect();
        workers.sort_by(|a, b| a.identifier.cmp(&b.identifier));

        for worker in workers {
            for device in &worker.devices {
                let health_status = if worker.is_healthy {
                    "Healthy"
                } else {
                    "Unhealthy"
                };
                let health_color = if worker.is_healthy {
                    Color::Green
                } else {
                    Color::Red
                };

                let cores_ratio = format!(
                    "{}/{}",
                    device.available_cuda_cores, device.total_cuda_cores
                );
                let memory_ratio = format!(
                    "{:.0}MB/{:.0}MB",
                    device.pod_memory_used as f64 / 1024.0 / 1024.0,
                    device.mem_limit as f64 / 1024.0 / 1024.0
                );
                let up_limit_str = format!("{}%", device.up_limit);

                // Truncate UUID for display
                let display_uuid = if device.uuid.len() > 12 {
                    format!("{}...", &device.uuid[..9])
                } else {
                    device.uuid.clone()
                };

                device_rows.push(Row::new(vec![
                    Cell::from(worker.identifier.as_str()),
                    Cell::from(display_uuid),
                    Cell::from(cores_ratio),
                    Cell::from(memory_ratio),
                    Cell::from(up_limit_str),
                    Cell::from(health_status).style(Style::default().fg(health_color)),
                ]));
            }
        }

        let total_devices: usize = self.workers.values().map(|w| w.devices.len()).sum();

        let widths = [
            Constraint::Length(15), // Worker
            Constraint::Length(15), // Device UUID
            Constraint::Length(18), // Available/Total Cores
            Constraint::Length(18), // Memory Used/Limit
            Constraint::Length(10), // Up Limit
            Constraint::Fill(1),    // Health
        ];

        let table = Table::new(device_rows, widths)
            .header(header)
            .block(Block::default().borders(Borders::ALL).title(format!(
                " Device Monitor ({} workers, {} devices) ",
                self.workers.len(),
                total_devices
            )))
            .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
            .highlight_symbol(">> ");

        frame.render_stateful_widget(table, area, &mut self.table_state);
    }
}

pub async fn run_tui_monitor(glob_pattern: String) -> Result<()> {
    let mut stdout = std::io::stdout();
    enable_raw_mode()?;
    stdout.execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut monitor = WorkerMonitor::new();

    let (tx, mut rx) = mpsc::channel(100);

    let pattern_clone = glob_pattern.clone();
    let tx_clone = tx.clone();
    let _watcher_task = tokio::spawn(async move {
        if let Err(e) = setup_file_watcher(&pattern_clone, tx_clone).await {
            tracing::error!("File watcher error: {}", e);
        }
    });

    let tx_clone = tx.clone();
    let _pattern_clone = glob_pattern.clone();
    let _refresh_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            if tx_clone.send(RefreshEvent::Timer).await.is_err() {
                break;
            }
        }
    });

    monitor.update_workers(&glob_pattern)?;

    let result = run_ui_loop(&mut terminal, &mut monitor, &mut rx, &glob_pattern).await;

    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;

    result
}

#[derive(Debug)]
enum RefreshEvent {
    FileChange,
    Timer,
}

async fn setup_file_watcher(pattern: &str, tx: mpsc::Sender<RefreshEvent>) -> Result<()> {
    let path = if pattern.starts_with("/dev/shm/") {
        "/dev/shm"
    } else {
        pattern.split('/').next().unwrap_or("/dev/shm")
    };

    let (watch_tx, mut watch_rx) = mpsc::channel(100);
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(event) = res {
                if watch_tx.blocking_send(event).is_err() {}
            }
        },
        notify::Config::default(),
    )?;

    watcher.watch(Path::new(path), RecursiveMode::NonRecursive)?;

    while let Some(_event) = watch_rx.recv().await {
        if tx.send(RefreshEvent::FileChange).await.is_err() {
            break;
        }
    }

    Ok(())
}

async fn run_ui_loop(
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
                            KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                            KeyCode::Down | KeyCode::Char('j') => monitor.next(),
                            KeyCode::Up | KeyCode::Char('k') => monitor.previous(),
                            KeyCode::Char('r') => {
                                if let Err(e) = monitor.update_workers(glob_pattern) {
                                    tracing::error!("Failed to update workers: {}", e);
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
                    Some(RefreshEvent::FileChange) | Some(RefreshEvent::Timer) => {
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
