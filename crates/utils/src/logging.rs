//! provides logging helpers

use std::env;
use std::path::Path;
use std::sync::OnceLock;
use tracing::level_filters::LevelFilter;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::RollingFileAppender;
use tracing_appender::rolling::Rotation;
use tracing_subscriber::fmt::layer;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Registry;

const DEFAULT_LOG_PREFIX: &str = "tf.log";
const ENABLE_LOG_ENV_VAR: &str = "TF_ENABLE_LOG";
const LOG_PATH_ENV_VAR: &str = "TF_LOG_PATH";
const LOG_LEVEL_ENV_VAR: &str = "TF_LOG_LEVEL";
const LOG_OFF: &str = "off";

static LOG_WORKER_GUARD: OnceLock<WorkerGuard> = OnceLock::new();

/// initiate the global tracing subscriber
pub fn get_fmt_layer() -> Box<dyn tracing_subscriber::Layer<Registry> + Send + Sync> {
    let filter = match env::var(ENABLE_LOG_ENV_VAR).as_deref() {
        Ok(LOG_OFF) | Ok("0") | Ok("false") => EnvFilter::new(LOG_OFF),
        _ => EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .with_env_var(LOG_LEVEL_ENV_VAR)
            .from_env_lossy(),
    };

    let fmt_layer = match env::var(LOG_PATH_ENV_VAR) {
        Ok(path) => {
            // path could be a specific a/b/c.log file name, split it to get base dir and prefix
            let path = Path::new(&path);
            let base_dir = path.parent().unwrap();
            let prefix = path.file_name().unwrap();
            let is_dir = path.is_dir();

            let appender = RollingFileAppender::builder()
                .rotation(Rotation::DAILY)
                .filename_prefix(if is_dir {
                    DEFAULT_LOG_PREFIX
                } else {
                    prefix.to_str().unwrap()
                })
                .max_log_files(7)
                .build(if is_dir { path } else { base_dir })
                .expect("failed to create rolling file appender");

            let (file_writer, guard) = tracing_appender::non_blocking(appender);

            // keep non blocking write thread alive in global scope
            LOG_WORKER_GUARD
                .set(guard)
                .expect("failed to set log worker guard");

            layer()
                .with_writer(file_writer)
                .with_target(true)
                .with_ansi(false)
                .boxed()
        }
        _ => layer()
            .with_writer(std::io::stdout)
            .with_target(true)
            .boxed(),
    };

    fmt_layer.with_filter(filter).boxed()
}

pub fn init() {
    let fmt_layer = get_fmt_layer();
    registry().with(fmt_layer).init();
}
