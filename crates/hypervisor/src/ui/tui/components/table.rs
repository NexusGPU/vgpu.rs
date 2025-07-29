use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Row, Table, TableState};
use std::collections::HashMap;

use crate::ui::tui::types::{AppState, WorkerInfo};

pub struct WorkerTable;

impl WorkerTable {
    pub fn render(
        workers: &HashMap<String, WorkerInfo>,
        table_state: &mut TableState,
        app_state: &AppState,
        frame: &mut Frame,
        area: Rect,
    ) {
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
        let mut workers_sorted: Vec<_> = workers.values().collect();
        workers_sorted.sort_by(|a, b| a.identifier.cmp(&b.identifier));

        for worker in workers_sorted {
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

        let total_devices: usize = workers.values().map(|w| w.devices.len()).sum();

        let widths = [
            Constraint::Min(15),    // Worker
            Constraint::Min(15),    // Device UUID
            Constraint::Length(18), // Available/Total Cores
            Constraint::Length(18), // Memory Used/Limit
            Constraint::Length(10), // Up Limit
            Constraint::Length(10), // Health
        ];

        let instructions = if matches!(app_state, AppState::Normal) {
            " ↑/↓: Navigate | Enter: Details | R: Refresh | Q: Quit "
        } else {
            " ESC: Close Dialog "
        };

        let table = Table::new(device_rows, widths)
            .header(header)
            .block(Block::default().borders(Borders::ALL).title(format!(
                " Device Monitor ({} workers, {} devices) {} ",
                workers.len(),
                total_devices,
                instructions
            )))
            .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
            .highlight_symbol(">> ");

        frame.render_stateful_widget(table, area, table_state);
    }
}
