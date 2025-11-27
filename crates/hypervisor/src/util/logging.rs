//! provides logging helpers

use std::env;
use std::fmt::{self};
use std::path::Path;

use std::path::PathBuf;

use tracing::field::Field;
use tracing::field::Visit;
use tracing::Event;
use tracing::Subscriber;
use tracing_appender::rolling::RollingFileAppender;
use tracing_appender::rolling::Rotation;
use tracing_subscriber::filter::FilterExt;
use tracing_subscriber::filter::{self};
use tracing_subscriber::fmt::format;
use tracing_subscriber::fmt::layer;
use tracing_subscriber::fmt::FmtContext;
use tracing_subscriber::fmt::FormatEvent;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry;
use utils::logging::LOG_PATH_ENV_VAR;

const DEFAULT_METRICS_PREFIX: &str = "metrics.log";

struct InfluxDBFormatter;

struct FieldVisitor {
    msg: String,
}

impl Visit for FieldVisitor {
    fn record_str(&mut self, _: &Field, value: &str) {
        self.msg.push_str(value);
    }

    fn record_debug(&mut self, _: &Field, value: &dyn fmt::Debug) {
        self.msg.push_str(&format!("{value:?}"));
    }
}

impl<S, N> FormatEvent<S, N> for InfluxDBFormatter
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> tracing_subscriber::fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let mut visitor = FieldVisitor { msg: String::new() };
        event.record(&mut visitor);
        write!(writer, "{}", visitor.msg)?;
        Ok(())
    }
}

/// initiate the global tracing subscriber
pub fn init<P: AsRef<Path>>(
    gpu_metrics_file: Option<P>,
) -> tracing_appender::non_blocking::WorkerGuard {
    let log_path = env::var(LOG_PATH_ENV_VAR).ok();
    let fmt_layer = utils::logging::get_fmt_layer(log_path);

    let gpu_metrics_file = gpu_metrics_file
        .map(|p| p.as_ref().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/logs/metrics.log"));
    let is_dir = gpu_metrics_file.is_dir();
    let (rotation_dir, prefix) = if is_dir {
        (gpu_metrics_file.as_path(), DEFAULT_METRICS_PREFIX)
    } else {
        let parent = gpu_metrics_file
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let prefix = gpu_metrics_file
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(DEFAULT_METRICS_PREFIX);
        (parent, prefix)
    };
    let env_filter = filter::EnvFilter::builder()
        .with_default_directive(filter::LevelFilter::INFO.into())
        .from_env_lossy();

    let fmt_layer = fmt_layer.with_filter(env_filter.and(filter::filter_fn(|metadata| {
        !metadata.target().eq("metrics")
    })));

    let (file_writer, file_guard) = match RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix(prefix)
        .max_log_files(3)
        .build(rotation_dir)
    {
        Ok(appender) => tracing_appender::non_blocking(appender),
        Err(err) => {
            tracing::error!(
                "failed to create metrics rolling file appender at {}: {err}; falling back to stdout",
                rotation_dir.display()
            );
            tracing_appender::non_blocking(std::io::stdout())
        }
    };

    let metrics_layer = layer()
        .event_format(InfluxDBFormatter {})
        .fmt_fields(format::DefaultFields::new())
        .with_writer(file_writer)
        .with_ansi(false)
        .with_filter(filter::filter_fn(|metadata| {
            metadata.target().eq("metrics")
        }));

    registry().with(fmt_layer).with(metrics_layer).init();
    file_guard
}
