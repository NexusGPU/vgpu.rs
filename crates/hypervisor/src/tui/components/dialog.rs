use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use std::time::{SystemTime, UNIX_EPOCH};

use super::utils::centered_rect;
use crate::tui::types::WorkerDetailedInfo;

pub struct DetailDialog;

impl DetailDialog {
    pub fn render(detailed_info: &WorkerDetailedInfo, frame: &mut Frame, area: Rect) {
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
            if detailed_info.is_healthy {
                "Healthy"
            } else {
                "Unhealthy"
            },
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
                    .title_style(Style::default().fg(Color::Yellow)),
            )
            .wrap(Wrap { trim: true })
            .style(Style::default().fg(Color::White));

        frame.render_widget(paragraph, popup_area);
    }
}
