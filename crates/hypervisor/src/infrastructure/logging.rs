//! provides logging helpers

use std::env;
use std::fmt::{self};
use std::path::Path;

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
        .as_ref()
        .map(|p| p.as_ref())
        .unwrap_or(Path::new("/logs/metrics.log"));

    let path = gpu_metrics_file.parent().expect("path");
    let file = gpu_metrics_file.file_name().expect("log file");
    let env_filter = filter::EnvFilter::builder()
        .with_default_directive(filter::LevelFilter::INFO.into())
        .from_env_lossy();

    let fmt_layer = fmt_layer.with_filter(env_filter.and(filter::filter_fn(|metadata| {
        !metadata.target().contains("metrics")
    })));

    let appender = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix(file.to_str().expect("metrics file name"))
        .max_log_files(3)
        .build(path)
        .expect("failed to create rolling file appender");

    let (file_writer, file_guard) = tracing_appender::non_blocking(appender);

    let metrics_layer = layer()
        .event_format(InfluxDBFormatter {})
        .fmt_fields(format::DefaultFields::new())
        .with_writer(file_writer)
        .with_ansi(false)
        .with_filter(filter::filter_fn(|metadata| {
            metadata.target().contains("metrics")
        }));

    registry().with(fmt_layer).with(metrics_layer).init();
    file_guard
}
