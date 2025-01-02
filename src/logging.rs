//! provides logging helpers

use tracing_appender::rolling;
use tracing_subscriber::filter::{self, FilterExt};
use tracing_subscriber::fmt::layer;
use tracing_subscriber::{prelude::*, registry};

/// initiate the global tracing subscriber
pub fn init() -> tracing_appender::non_blocking::WorkerGuard {
    let env_filter = filter::EnvFilter::builder()
        .with_default_directive(filter::LevelFilter::INFO.into())
        .from_env_lossy();

    let fmt_layer = layer()
        .with_writer(std::io::stdout)
        .with_target(true)
        .with_filter(env_filter.and(filter::filter_fn(|metadata| {
            !metadata.target().contains("metrics")
        })));

    let file_appender = rolling::never("logs", "metrics.log");
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
