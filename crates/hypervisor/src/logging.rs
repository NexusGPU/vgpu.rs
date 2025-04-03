//! provides logging helpers

use std::collections::HashMap;
use std::fmt::{self};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{
    field::{Field, Visit},
    Event, Subscriber,
};
use tracing_subscriber::fmt::FormatEvent;

use tracing_appender::rolling;
use tracing_subscriber::filter::{self, FilterExt};
use tracing_subscriber::fmt::layer;
use tracing_subscriber::{prelude::*, registry};

struct InfluxDBFormatter;

struct FieldVisitor<'a> {
    tags: HashMap<&'a str, String>,
    fields: HashMap<&'a str, String>,
}

impl Visit for FieldVisitor<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name().starts_with("tag_") {
            self.tags.insert(&field.name()[4..], value.to_string());
        } else {
            self.fields.insert(field.name(), value.to_string());
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name().starts_with("tag_") {
            self.tags.insert(&field.name()[4..], format!("{:?}", value));
        } else {
            self.fields.insert(field.name(), format!("{:?}", value));
        }
    }
}

impl<S, N> FormatEvent<S, N> for InfluxDBFormatter
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> tracing_subscriber::fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: tracing_subscriber::fmt::format::Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let mut visitor = FieldVisitor {
            tags: HashMap::new(),
            fields: HashMap::new(),
        };
        event.record(&mut visitor);

        // Get measurement name from target
        let measurement = event.metadata().target();
        write!(writer, "{}", measurement.strip_prefix("metrics.").unwrap())?;

        // Write all tags
        for (key, value) in visitor.tags.iter() {
            write!(writer, ",{}={}", key, value)?;
        }

        // Write fields
        write!(writer, " ")?;
        let mut first = true;
        for (key, value) in visitor.fields.iter() {
            if !first {
                write!(writer, ",")?;
            }
            write!(writer, "{}={}", key, value)?;
            first = false;
        }

        // Write timestamp in nanoseconds
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        writeln!(writer, " {}", timestamp)?;

        Ok(())
    }
}

/// initiate the global tracing subscriber
pub(crate) fn init<P: AsRef<Path>>(
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
        .event_format(InfluxDBFormatter {})
        .fmt_fields(tracing_subscriber::fmt::format::DefaultFields::new())
        .with_writer(file_writer)
        .with_ansi(false)
        .with_filter(filter::filter_fn(|metadata| {
            metadata.target().contains("metrics")
        }));

    registry().with(fmt_layer).with(metrics_layer).init();
    file_guard
}
