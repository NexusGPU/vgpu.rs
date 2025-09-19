use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::tui::types::ShmEntry;

pub struct ShmDetailDialog;

impl ShmDetailDialog {
    pub fn render(entry: &ShmEntry, frame: &mut Frame, area: Rect) {
        let popup_area = centered_rect(80, 70, area);
        frame.render_widget(Clear, popup_area);

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let heartbeat_age = if entry.last_heartbeat > 0 {
            current_time.saturating_sub(entry.last_heartbeat)
        } else {
            0
        };

        let mut content_lines = vec![
            vec![
                "Namespace: ".into(),
                entry.pod_identifier.namespace.clone().dim(),
            ]
            .into(),
            vec!["Pod Name: ".into(), entry.pod_identifier.name.clone().dim()].into(),
            "".into(),
            vec![
                "Health: ".into(),
                if entry.is_healthy {
                    "Healthy".green()
                } else {
                    "Unhealthy".red()
                },
            ]
            .into(),
            vec!["Version: ".into(), format!("v{}", entry.version).into()].into(),
            vec![
                "Last Heartbeat: ".into(),
                if entry.last_heartbeat > 0 {
                    format!("{} ({} seconds ago)", entry.last_heartbeat, heartbeat_age).into()
                } else {
                    "Never".dim()
                },
            ]
            .into(),
            vec![
                "Device Count: ".into(),
                format!("{}", entry.device_count).into(),
            ]
            .into(),
            vec![
                "Active PIDs: ".into(),
                format!("{}", entry.active_pids.len()).into(),
            ]
            .into(),
            "".into(),
            "=== ACTIVE PIDs ===".bold().into(),
        ];

        if entry.active_pids.is_empty() {
            content_lines.push("No active PIDs".dim().into());
        } else {
            for (i, pid) in entry.active_pids.iter().enumerate() {
                content_lines
                    .push(vec![format!("PID {}: ", i + 1).into(), format!("{pid}").into()].into());
            }
        }

        content_lines.push("".into());
        content_lines.push("=== DEVICES ===".bold().into());

        if entry.devices.is_empty() {
            content_lines.push("No devices".dim().into());
        } else {
            for device in entry.devices.iter() {
                content_lines.push("".into());
                content_lines.push(
                    vec![
                        "Device ".into(),
                        format!("{}", device.device_index).bold(),
                        ":".into(),
                    ]
                    .into(),
                );
                content_lines.push(vec!["• UUID: ".into(), device.uuid.clone().dim()].into());
                content_lines.push(
                    vec![
                        "• Available Cores: ".into(),
                        format!("{}", device.available_cores).into(),
                        " / ".dim(),
                        format!("{}", device.total_cores).into(),
                        format!(
                            " ({:.1}%)",
                            if device.total_cores > 0 {
                                (device.available_cores as f64 / device.total_cores as f64) * 100.0
                            } else {
                                0.0
                            }
                        )
                        .dim(),
                    ]
                    .into(),
                );
                content_lines.push(
                    vec![
                        "• Memory Used: ".into(),
                        format!("{:.1} MB", device.pod_memory_used as f64 / 1024.0 / 1024.0).into(),
                        " / ".dim(),
                        format!("{:.1} MB", device.mem_limit as f64 / 1024.0 / 1024.0).into(),
                        format!(
                            " ({:.1}%)",
                            if device.mem_limit > 0 {
                                (device.pod_memory_used as f64 / device.mem_limit as f64) * 100.0
                            } else {
                                0.0
                            }
                        )
                        .dim(),
                    ]
                    .into(),
                );
                content_lines.push(
                    vec![
                        "• Up Limit: ".into(),
                        format!("{}%", device.up_limit).into(),
                    ]
                    .into(),
                );
                content_lines.push(
                    vec![
                        "• Status: ".into(),
                        if device.is_active {
                            "Active".green()
                        } else {
                            "Inactive".red()
                        },
                    ]
                    .into(),
                );
            }
        }

        content_lines.push("".into());
        content_lines.push("Press ESC to close this dialog".dim().into());

        let paragraph = Paragraph::new(content_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" Shared Memory Details: {} ", entry.pod_identifier))
                    .bold()
                    .cyan(),
            )
            .wrap(Wrap { trim: true });

        frame.render_widget(paragraph, popup_area);
    }
}

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
