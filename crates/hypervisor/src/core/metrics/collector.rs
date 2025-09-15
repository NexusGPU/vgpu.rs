//! Metrics collector implementation

// Re-export from platform implementation if it exists
// This is a placeholder for the metrics collection logic
pub struct MetricsCollector;

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self
    }
}
