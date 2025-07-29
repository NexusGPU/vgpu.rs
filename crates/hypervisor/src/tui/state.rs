use anyhow::{Context, Result};
use glob::glob;
use ratatui::prelude::*;
use ratatui::widgets::TableState;
use std::collections::HashMap;
use utils::shared_memory::handle::SharedMemoryHandle;

use crate::tui::components::{DetailDialog, WorkerTable};
use crate::tui::types::{AppState, DeviceInfo, WorkerDetailedInfo, WorkerInfo};

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

            match self.read_worker_info(&identifier) {
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

    fn read_worker_info(&self, identifier: &str) -> Result<WorkerInfo> {
        let handle = SharedMemoryHandle::open(identifier)?;
        let state = handle.get_state();
        let is_healthy = state.is_healthy(10);

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
            )) = state.get_device_info(i)
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
        let handle = SharedMemoryHandle::open(identifier)?;
        let state = handle.get_state();
        let is_healthy = state.is_healthy(10);
        let (last_heartbeat, active_pids, version) = state.get_detailed_state_info();

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
            )) = state.get_device_info(i)
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

    pub fn app_state(&self) -> &AppState {
        &self.app_state
    }

    pub fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();

        // Render main table
        WorkerTable::render(
            &self.workers,
            &mut self.table_state,
            &self.app_state,
            frame,
            area,
        );

        // Render detail dialog if in DetailDialog state
        if let AppState::DetailDialog(ref detailed_info) = self.app_state {
            DetailDialog::render(detailed_info, frame, area);
        }
    }
}

/// Mock implementation of WorkerMonitor for testing and local development
pub struct MockWorkerMonitor {
    workers: HashMap<String, WorkerInfo>,
    table_state: TableState,
    selected_index: usize,
    app_state: AppState,
    mock_counter: u64,
}

impl MockWorkerMonitor {
    pub fn new() -> Self {
        Self {
            workers: HashMap::new(),
            table_state: TableState::default(),
            selected_index: 0,
            app_state: AppState::Normal,
            mock_counter: 0,
        }
    }

    /// Generate mock worker data
    pub fn update_mock_workers(&mut self) {
        self.mock_counter += 1;
        let mut new_workers = HashMap::new();

        // Create 3-5 mock workers
        let worker_count = 3 + (self.mock_counter % 3) as usize;

        for i in 0..worker_count {
            let worker_id = format!("worker-{:02}", i + 1);
            let device_count = 1 + (i % 2); // 1-2 devices per worker

            let mut devices = Vec::new();
            for j in 0..device_count {
                let base_cores = 2048 + (i * 512) as u32;
                let used_cores =
                    (self.mock_counter * 10 + (i * j * 50) as u64) % (base_cores as u64 / 2);
                let available_cores = (base_cores as i32) - (used_cores as i32);

                let base_memory = 8_000_000_000u64 + (i as u64 * 2_000_000_000);
                let used_memory = (self.mock_counter * 100_000_000 + (i * j * 500_000_000) as u64)
                    % (base_memory / 2);

                devices.push(DeviceInfo {
                    uuid: format!(
                        "GPU-{:08x}-{:04x}-{:04x}",
                        0x12345678 + (i * 0x1000) as u32,
                        0x1234 + (j * 0x100) as u16,
                        0x5678 + (i * j * 0x10) as u16
                    ),
                    available_cuda_cores: available_cores,
                    total_cuda_cores: base_cores,
                    mem_limit: base_memory,
                    pod_memory_used: used_memory,
                    up_limit: 80 + ((self.mock_counter + i as u64) % 20) as u32, // 80-99%
                });
            }

            // Simulate health status - occasionally make a worker unhealthy
            let is_healthy = (self.mock_counter + i as u64) % 7 != 0;

            new_workers.insert(
                worker_id.clone(),
                WorkerInfo {
                    identifier: worker_id,
                    devices,
                    is_healthy,
                },
            );
        }

        self.workers = new_workers;
        self.update_selection();
    }

    /// Generate mock detailed worker info
    fn generate_mock_detailed_info(&self, identifier: &str) -> WorkerDetailedInfo {
        use std::time::{SystemTime, UNIX_EPOCH};

        let worker_info = self.workers.get(identifier).unwrap();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Generate some mock PIDs
        let active_pids = vec![
            1000 + (identifier.len() * 100),
            2000 + (identifier.len() * 200),
            3000 + (identifier.len() * 300),
        ];

        WorkerDetailedInfo {
            identifier: identifier.to_string(),
            devices: worker_info.devices.clone(),
            is_healthy: worker_info.is_healthy,
            last_heartbeat: current_time - (self.mock_counter % 30), // 0-30 seconds ago
            active_pids,
            version: 42 + (identifier.len() as u32 % 10),
            device_count: worker_info.devices.len(),
        }
    }

    pub fn show_details(&mut self) {
        if let Some(selected_worker) = self.get_selected_worker_identifier() {
            let detailed_info = self.generate_mock_detailed_info(&selected_worker);
            self.app_state = AppState::DetailDialog(detailed_info);
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

    pub fn app_state(&self) -> &AppState {
        &self.app_state
    }

    pub fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();

        // Render main table
        WorkerTable::render(
            &self.workers,
            &mut self.table_state,
            &self.app_state,
            frame,
            area,
        );

        // Render detail dialog if in DetailDialog state
        if let AppState::DetailDialog(ref detailed_info) = self.app_state {
            DetailDialog::render(detailed_info, frame, area);
        }
    }
}
