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
pub struct WorkerDetailedInfo {
    pub identifier: String,
    pub devices: Vec<DeviceInfo>,
    pub is_healthy: bool,
    pub last_heartbeat: u64,
    pub active_pids: Vec<usize>,
    pub version: u32,
    pub device_count: usize,
}

#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub identifier: String,
    pub devices: Vec<DeviceInfo>,
    pub is_healthy: bool,
}

#[derive(Debug, Clone)]
enum AppState {
    Normal,
    DetailDialog(WorkerDetailedInfo),
}

pub struct WorkerMonitor {
    workers: HashMap<String, WorkerInfo>,
    table_state: TableState,
    selected_index: usize,
    app_state: AppState,
}

impl WorkerMonitor {
    pub fn new() -> Self {
        Self {
            workers: HashMap::new(),
            table_state: TableState::default(),
            selected_index: 0,
            app_state: AppState::Normal,
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
        let is_healthy = state.is_healthy(10); // 10 seconds timeout

        // Read all active devices
        let mut devices = Vec::new();
        let device_count = state.device_count();

        for i in 0..device_count {
            if let Some((
                uuid,
                available_cores,
                total_cores,
                mem_limit,
                pod_memory_used,
                up_limit,
                is_active,
            )) = state.get_complete_device_info(i)
            {
                if is_active {
                    devices.push(DeviceInfo {
                        uuid,
                        available_cuda_cores: available_cores,
                        total_cuda_cores: total_cores,
                        mem_limit,
                        pod_memory_used,
                        up_limit,
                    });
                }
            }
        }

        Ok(WorkerInfo {
            identifier: identifier.to_string(),
            devices,
            is_healthy,
        })
    }

    fn read_detailed_worker_info(&self, identifier: &str) -> Result<WorkerDetailedInfo> {
        let handle =
            SharedMemoryHandle::open(identifier).context("Failed to open shared memory")?;

        let state = handle.get_state();
        let is_healthy = state.is_healthy(10);
        let (last_heartbeat, active_pids, version) = state.get_detailed_state_info();

        // Read all active devices
        let mut devices = Vec::new();
        let device_count = state.device_count();

        for i in 0..device_count {
            if let Some((
                uuid,
                available_cores,
                total_cores,
                mem_limit,
                pod_memory_used,
                up_limit,
                is_active,
            )) = state.get_complete_device_info(i)
            {
                if is_active {
                    devices.push(DeviceInfo {
                        uuid,
                        available_cuda_cores: available_cores,
                        total_cuda_cores: total_cores,
                        mem_limit,
                        pod_memory_used,
                        up_limit,
                    });
                }
            }
        }

        Ok(WorkerDetailedInfo {
            identifier: identifier.to_string(),
            devices,
            is_healthy,
            last_heartbeat,
            active_pids,
            version,
            device_count,
        })
    }

    pub fn show_details(&mut self) {
        if let Some(selected_worker) = self.get_selected_worker_identifier() {
            match self.read_detailed_worker_info(&selected_worker) {
                Ok(detailed_info) => {
                    self.app_state = AppState::DetailDialog(detailed_info);
                }
                Err(e) => {
                    tracing::error!("Failed to read detailed worker info: {}", e);
                }
            }
        }
    }

    pub fn close_details(&mut self) {
        self.app_state = AppState::Normal;
    }

    fn get_selected_worker_identifier(&self) -> Option<String> {
        let workers: Vec<_> = self.workers.keys().cloned().collect();
        if self.selected_index < workers.len() {
            Some(workers[self.selected_index].clone())
        } else {
            None
        }
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

        // Render main table
        self.render_main_table(frame, area);

        // Render detail dialog if in DetailDialog state
        if let AppState::DetailDialog(ref detailed_info) = self.app_state {
            self.render_detail_dialog(frame, area, detailed_info);
        }
    }

    fn render_main_table(&mut self, frame: &mut Frame, area: Rect) {
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

                device_rows.push(Row::new(vec![
                    Cell::from(worker.identifier.as_str()),
                    Cell::from(device.uuid.as_str()),
                    Cell::from(cores_ratio),
                    Cell::from(memory_ratio),
                    Cell::from(up_limit_str),
                    Cell::from(health_status).style(Style::default().fg(health_color)),
                ]));
            }
        }

        let total_devices: usize = self.workers.values().map(|w| w.devices.len()).sum();

        let widths = [
            Constraint::Min(15),    // Worker
            Constraint::Min(15),    // Device UUID
            Constraint::Length(18), // Available/Total Cores
            Constraint::Length(18), // Memory Used/Limit
            Constraint::Length(10), // Up Limit
            Constraint::Length(10), // Health
        ];

        let instructions = if matches!(self.app_state, AppState::Normal) {
            " ↑/↓: Navigate | Enter: Details | R: Refresh | Q: Quit "
        } else {
            " ESC: Close Dialog "
        };

        let table = Table::new(device_rows, widths)
            .header(header)
            .block(Block::default().borders(Borders::ALL).title(format!(
                " Device Monitor ({} workers, {} devices) {} ",
                self.workers.len(),
                total_devices,
                instructions
            )))
            .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
            .highlight_symbol(">> ");

        frame.render_stateful_widget(table, area, &mut self.table_state);
    }

    fn render_detail_dialog(&self, frame: &mut Frame, area: Rect, detailed_info: &WorkerDetailedInfo) {
        use ratatui::widgets::{Clear, Paragraph, Wrap};
        use std::time::{SystemTime, UNIX_EPOCH};

        // Create a centered popup area
        let popup_area = centered_rect(80, 70, area);

        // Clear the area
        frame.render_widget(Clear, popup_area);

        // Format the detailed information
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let heartbeat_age = if detailed_info.last_heartbeat > 0 {
            current_time.saturating_sub(detailed_info.last_heartbeat)
        } else {
            0
        };

        let mut content = format!(
            "Worker: {}\n\
            Version: v{}\n\
            Health: {}\n\
            Last Heartbeat: {} ({} seconds ago)\n\
            Active PIDs: {} PIDs\n\
            Device Count: {}\n\n\
            === ACTIVE PIDs ===\n",
            detailed_info.identifier,
            detailed_info.version,
            if detailed_info.is_healthy { "Healthy" } else { "Unhealthy" },
            if detailed_info.last_heartbeat > 0 {
                format!("{}", detailed_info.last_heartbeat)
            } else {
                "Never".to_string()
            },
            heartbeat_age,
            detailed_info.active_pids.len(),
            detailed_info.device_count
        );

        // Add PIDs information
        if detailed_info.active_pids.is_empty() {
            content.push_str("No active PIDs\n");
        } else {
            for (i, pid) in detailed_info.active_pids.iter().enumerate() {
                content.push_str(&format!("PID {}: {}\n", i + 1, pid));
            }
        }

        content.push_str("\n=== DEVICES ===\n");

        // Add device details
        if detailed_info.devices.is_empty() {
            content.push_str("No active devices\n");
        } else {
            for (i, device) in detailed_info.devices.iter().enumerate() {
                content.push_str(&format!(
                    "\nDevice {}:\n\
                    • UUID: {}\n\
                    • Available Cores: {} / {} ({:.1}%)\n\
                    • Memory Used: {:.1} MB / {:.1} MB ({:.1}%)\n\
                    • Up Limit: {}%\n",
                    i + 1,
                    device.uuid,
                    device.available_cuda_cores,
                    device.total_cuda_cores,
                    (device.available_cuda_cores as f64 / device.total_cuda_cores as f64) * 100.0,
                    device.pod_memory_used as f64 / 1024.0 / 1024.0,
                    device.mem_limit as f64 / 1024.0 / 1024.0,
                    if device.mem_limit > 0 {
                        (device.pod_memory_used as f64 / device.mem_limit as f64) * 100.0
                    } else {
                        0.0
                    },
                    device.up_limit
                ));
            }
        }

        content.push_str("\n\nPress ESC to close this dialog");

        let paragraph = Paragraph::new(content)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" Worker Details: {} ", detailed_info.identifier))
                    .title_style(Style::default().fg(Color::Yellow))
            )
            .wrap(Wrap { trim: true })
            .style(Style::default().fg(Color::White));

        frame.render_widget(paragraph, popup_area);
    }
}

/// Helper function to create a centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Mock data generator for TUI development
fn generate_mock_workers() -> HashMap<String, WorkerInfo> {
    use std::collections::HashMap;
    
    let mut workers = HashMap::new();
    
    // Mock worker 1 - Healthy with multiple devices
    workers.insert("pod_example_123".to_string(), WorkerInfo {
        identifier: "pod_example_123".to_string(),
        devices: vec![
            DeviceInfo {
                uuid: "GPU-12345678-1234-1234-1234-123456789012".to_string(),
                available_cuda_cores: 1024,
                total_cuda_cores: 2048,
                mem_limit: 1024 * 1024 * 1024, // 1GB
                pod_memory_used: 512 * 1024 * 1024, // 512MB
                up_limit: 80,
            },
            DeviceInfo {
                uuid: "GPU-87654321-4321-4321-4321-210987654321".to_string(),
                available_cuda_cores: 512,
                total_cuda_cores: 1024,
                mem_limit: 512 * 1024 * 1024, // 512MB
                pod_memory_used: 256 * 1024 * 1024, // 256MB
                up_limit: 90,
            },
        ],
        is_healthy: true,
    });
    
    // Mock worker 2 - Unhealthy with high utilization
    workers.insert("pod_stressed_456".to_string(), WorkerInfo {
        identifier: "pod_stressed_456".to_string(),
        devices: vec![
            DeviceInfo {
                uuid: "GPU-11111111-2222-3333-4444-555555555555".to_string(),
                available_cuda_cores: 64,
                total_cuda_cores: 2048,
                mem_limit: 2048 * 1024 * 1024, // 2GB
                pod_memory_used: 1900 * 1024 * 1024, // 1.9GB (high usage)
                up_limit: 70,
            },
        ],
        is_healthy: false,
    });
    
    // Mock worker 3 - Normal with single device
    workers.insert("pod_normal_789".to_string(), WorkerInfo {
        identifier: "pod_normal_789".to_string(),
        devices: vec![
            DeviceInfo {
                uuid: "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee".to_string(),
                available_cuda_cores: 1500,
                total_cuda_cores: 2048,
                mem_limit: 4096 * 1024 * 1024, // 4GB
                pod_memory_used: 1024 * 1024 * 1024, // 1GB
                up_limit: 85,
            },
        ],
        is_healthy: true,
    });
    
    // Mock worker 4 - Zero utilization
    workers.insert("pod_idle_999".to_string(), WorkerInfo {
        identifier: "pod_idle_999".to_string(),
        devices: vec![
            DeviceInfo {
                uuid: "GPU-00000000-0000-0000-0000-000000000000".to_string(),
                available_cuda_cores: 2048,
                total_cuda_cores: 2048,
                mem_limit: 8192 * 1024 * 1024, // 8GB
                pod_memory_used: 0, // No usage
                up_limit: 95,
            },
        ],
        is_healthy: true,
    });
    
    workers
}

/// Generate mock detailed worker info
fn generate_mock_detailed_worker(identifier: &str) -> WorkerDetailedInfo {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let current_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    match identifier {
        "pod_example_123" => WorkerDetailedInfo {
            identifier: identifier.to_string(),
            devices: vec![
                DeviceInfo {
                    uuid: "GPU-12345678-1234-1234-1234-123456789012".to_string(),
                    available_cuda_cores: 1024,
                    total_cuda_cores: 2048,
                    mem_limit: 1024 * 1024 * 1024,
                    pod_memory_used: 512 * 1024 * 1024,
                    up_limit: 80,
                },
                DeviceInfo {
                    uuid: "GPU-87654321-4321-4321-4321-210987654321".to_string(),
                    available_cuda_cores: 512,
                    total_cuda_cores: 1024,
                    mem_limit: 512 * 1024 * 1024,
                    pod_memory_used: 256 * 1024 * 1024,
                    up_limit: 90,
                },
            ],
            is_healthy: true,
            last_heartbeat: current_time - 15, // 15 seconds ago
            active_pids: vec![12345, 12346, 12347],
            version: 1,
            device_count: 2,
        },
        "pod_stressed_456" => WorkerDetailedInfo {
            identifier: identifier.to_string(),
            devices: vec![
                DeviceInfo {
                    uuid: "GPU-11111111-2222-3333-4444-555555555555".to_string(),
                    available_cuda_cores: 64,
                    total_cuda_cores: 2048,
                    mem_limit: 2048 * 1024 * 1024,
                    pod_memory_used: 1900 * 1024 * 1024,
                    up_limit: 70,
                },
            ],
            is_healthy: false,
            last_heartbeat: current_time - 120, // 2 minutes ago (stale)
            active_pids: vec![98765, 98766, 98767, 98768, 98769], // More PIDs
            version: 1,
            device_count: 1,
        },
        "pod_normal_789" => WorkerDetailedInfo {
            identifier: identifier.to_string(),
            devices: vec![
                DeviceInfo {
                    uuid: "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee".to_string(),
                    available_cuda_cores: 1500,
                    total_cuda_cores: 2048,
                    mem_limit: 4096 * 1024 * 1024,
                    pod_memory_used: 1024 * 1024 * 1024,
                    up_limit: 85,
                },
            ],
            is_healthy: true,
            last_heartbeat: current_time - 5, // 5 seconds ago
            active_pids: vec![55555],
            version: 1,
            device_count: 1,
        },
        "pod_idle_999" => WorkerDetailedInfo {
            identifier: identifier.to_string(),
            devices: vec![
                DeviceInfo {
                    uuid: "GPU-00000000-0000-0000-0000-000000000000".to_string(),
                    available_cuda_cores: 2048,
                    total_cuda_cores: 2048,
                    mem_limit: 8192 * 1024 * 1024,
                    pod_memory_used: 0,
                    up_limit: 95,
                },
            ],
            is_healthy: true,
            last_heartbeat: current_time - 1, // 1 second ago
            active_pids: vec![], // No active PIDs
            version: 1,
            device_count: 1,
        },
        _ => WorkerDetailedInfo {
            identifier: identifier.to_string(),
            devices: vec![],
            is_healthy: false,
            last_heartbeat: 0,
            active_pids: vec![],
            version: 1,
            device_count: 0,
        },
    }
}

/// Mock TUI monitor for development and testing
pub async fn run_tui_monitor_mock() -> Result<()> {
    let mut stdout = std::io::stdout();
    enable_raw_mode()?;
    stdout.execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut monitor = WorkerMonitor::new();
    
    // Load mock data
    monitor.workers = generate_mock_workers();
    monitor.update_selection();

    let (tx, mut rx) = mpsc::channel(100);

    // Mock refresh task that updates some data periodically
    let tx_clone = tx.clone();
    let _refresh_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;
            if tx_clone.send(RefreshEvent::Timer).await.is_err() {
                break;
            }
        }
    });

    tracing::info!("Started TUI monitor in MOCK mode");

    let result = run_mock_ui_loop(&mut terminal, &mut monitor, &mut rx).await;

    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;

    result
}

#[derive(Debug)]
enum RefreshEvent {
    FileChange,
    Timer,
}

impl WorkerMonitor {
    /// Show details using mock data
    pub fn show_details_mock(&mut self) {
        if let Some(selected_worker) = self.get_selected_worker_identifier() {
            let detailed_info = generate_mock_detailed_worker(&selected_worker);
            self.app_state = AppState::DetailDialog(detailed_info);
        }
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
                if let Err(e) = watch_tx.blocking_send(event) {
                    tracing::error!("Failed to send event to watch_tx, error: {e:?}");
                }
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
                            KeyCode::Char('q') | KeyCode::Esc => {
                                match monitor.app_state {
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
                                if matches!(monitor.app_state, AppState::Normal) {
                                    monitor.next();
                                }
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                if matches!(monitor.app_state, AppState::Normal) {
                                    monitor.previous();
                                }
                            }
                            KeyCode::Enter => {
                                if matches!(monitor.app_state, AppState::Normal) {
                                    monitor.show_details();
                                }
                            }
                            KeyCode::Char('r') => {
                                if matches!(monitor.app_state, AppState::Normal) {
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

async fn run_mock_ui_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    monitor: &mut WorkerMonitor,
    rx: &mut mpsc::Receiver<RefreshEvent>,
) -> Result<()> {
    loop {
        terminal.draw(|f| monitor.render(f))?;

        tokio::select! {
            event = tokio::time::timeout(Duration::from_millis(100), async { event::read() }) => {
                match event {
                    Ok(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => {
                                match monitor.app_state {
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
                                if matches!(monitor.app_state, AppState::Normal) {
                                    monitor.next();
                                }
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                if matches!(monitor.app_state, AppState::Normal) {
                                    monitor.previous();
                                }
                            }
                            KeyCode::Enter => {
                                if matches!(monitor.app_state, AppState::Normal) {
                                    monitor.show_details_mock();
                                }
                            }
                            KeyCode::Char('r') => {
                                if matches!(monitor.app_state, AppState::Normal) {
                                    // Refresh mock data
                                    monitor.workers = generate_mock_workers();
                                    monitor.update_selection();
                                    tracing::info!("Refreshed mock data");
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
                    Some(RefreshEvent::Timer) => {
                        // Periodically refresh mock data with slight variations
                        monitor.workers = generate_mock_workers();
                        monitor.update_selection();
                    }
                    Some(RefreshEvent::FileChange) => {
                        // Not applicable in mock mode
                    }
                    None => return Ok(()),
                }
            }
        }
    }
}
