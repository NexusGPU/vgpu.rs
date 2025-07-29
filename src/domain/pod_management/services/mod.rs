//! Service layer for pod management

pub mod device_service;
pub mod worker_service;
pub mod resource_monitor;
pub mod metrics;

// Re-export main services
pub use device_service::DeviceService;
pub use worker_service::WorkerService;
pub use resource_monitor::{ResourceMonitor, MetricsCollector, ConsoleMetricsCollector};
pub use metrics::{PrometheusMetricsCollector, init_metrics, get_metrics};