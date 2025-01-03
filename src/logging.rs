//! provides logging helpers

use std::path::Path;

use tracing_appender::rolling;
use tracing_subscriber::filter::{self, FilterExt};
use tracing_subscriber::fmt::layer;
use tracing_subscriber::{prelude::*, registry};

/// initiate the global tracing subscriber
pub fn init<P: AsRef<Path>>(
    gpu_metrics_file: Option<P>,
) -> tracing_appender::non_blocking::WorkerGuard {
    let gpu_metrics_file = gpu_metrics_file
        .as_ref()
        .map(|p| p.as_ref())
        .unwrap_or(Path::new("logs/metrics.log"));

    let path = gpu_metrics_file.parent().expect("path");
    let file = gpu_metrics_file.file_name().expect("log file");
    let env_filter = filter::EnvFilter::builder()
        .with_default_directive(filter::LevelFilter::INFO.into())
        .from_env_lossy();

    let fmt_layer = layer()
        .with_writer(std::io::stdout)
        .with_target(true)
        .with_filter(env_filter.and(filter::filter_fn(|metadata| {
            !metadata.target().contains("metrics")
        })));

    let file_appender = rolling::never(path, file);
    let (file_writer, file_guard) = tracing_appender::non_blocking(file_appender);

    let metrics_layer = layer()
        .with_writer(file_writer)
        .with_ansi(false)
        .with_target(false)
        .with_filter(filter::filter_fn(|metadata| {
            metadata.target().contains("metrics")
        }));

    registry().with(fmt_layer).with(metrics_layer).init();
    file_guard
}
