use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Row, Table, TableState};

use crate::tui::types::{AppState, ShmEntry};

pub struct ShmTable;

impl ShmTable {
    pub fn render(
        entries: &[ShmEntry],
        table_state: &mut TableState,
        app_state: &AppState,
        frame: &mut Frame,
        area: Rect,
    ) {
        let header_cells = [
            "Pod",
            "Devices",
            "Health",
            "Version",
            "PIDs",
            "Last Heartbeat",
        ]
        .iter()
        .map(|h| Cell::from(*h).bold().cyan());

        let header = Row::new(header_cells).height(1).bottom_margin(1);

        let mut rows = Vec::new();
        for entry in entries {
            let health_text = if entry.is_healthy {
                "Healthy"
            } else {
                "Unhealthy"
            };
            let health_color = if entry.is_healthy {
                Color::Green
            } else {
                Color::Red
            };

            let heartbeat_text = if entry.last_heartbeat > 0 {
                format!("{}", entry.last_heartbeat)
            } else {
                "Never".into()
            };

            rows.push(Row::new(vec![
                Cell::from(entry.pod_identifier.to_string()),
                Cell::from(format!("{}", entry.device_count)),
                Cell::from(health_text).fg(health_color),
                Cell::from(format!("v{}", entry.version)),
                Cell::from(format!("{}", entry.active_pids.len())),
                Cell::from(heartbeat_text).dim(),
            ]));
        }

        let widths = [
            Constraint::Min(20),    // Pod
            Constraint::Length(8),  // Devices
            Constraint::Length(10), // Health
            Constraint::Length(8),  // Version
            Constraint::Length(6),  // PIDs
            Constraint::Min(15),    // Last Heartbeat
        ];

        let instructions = if matches!(app_state, AppState::Normal) {
            " ↑/↓: Navigate | Enter: Details | R: Refresh | Q: Quit "
        } else {
            " ESC: Close Dialog "
        };

        let table = Table::new(rows, widths)
            .header(header)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(
                        " Shared Memory Monitor ({} entries) {} ",
                        entries.len(),
                        instructions
                    ))
                    .bold(),
            )
            .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
            .highlight_symbol(">> ");

        frame.render_stateful_widget(table, area, table_state);
    }
}
