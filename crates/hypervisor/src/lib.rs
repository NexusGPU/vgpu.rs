mod api;
pub mod app;
pub mod config;
mod domain;
mod infrastructure;
pub mod tui;

// Re-export main modules
pub use domain::hypervisor;
pub use domain::pod_management;
pub use domain::process;
pub use domain::scheduler;
pub use infrastructure::gpu_device_state_watcher;
pub use infrastructure::gpu_init;
pub use infrastructure::gpu_observer;
pub use infrastructure::host_pid_probe;
pub use infrastructure::k8s;
pub use infrastructure::kube_client;
pub use infrastructure::limiter_comm;
pub use infrastructure::logging;
pub use infrastructure::metrics;
