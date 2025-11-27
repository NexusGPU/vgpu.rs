//! provides logging helpers

use std::env;
use std::path::Path;

use tracing::level_filters::LevelFilter;
use tracing_appender::rolling::RollingFileAppender;
use tracing_appender::rolling::Rotation;
use tracing_subscriber::fmt::layer;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry;
use tracing_subscriber::{EnvFilter, Layer, Registry};

const DEFAULT_LOG_PREFIX: &str = "tf.log";
const ENABLE_LOG_ENV_VAR: &str = "TF_ENABLE_LOG";
pub const LOG_PATH_ENV_VAR: &str = "TF_LOG_PATH";
const LOG_LEVEL_ENV_VAR: &str = "TF_LOG_LEVEL";
const LOG_OFF: &str = "off";

/// initiate the global tracing subscriber
pub fn get_fmt_layer(log_path: Option<String>) -> Box<dyn Layer<Registry> + Send + Sync> {
    let filter = match env::var(ENABLE_LOG_ENV_VAR).as_deref() {
        Ok(LOG_OFF) | Ok("0") | Ok("false") => EnvFilter::new(LOG_OFF),
        _ => EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .with_env_var(LOG_LEVEL_ENV_VAR)
            .from_env_lossy(),
    };

    let fmt_layer = match log_path {
        Some(path) => {
            // path could be a specific a/b/c.log file name, split it to get base dir and prefix
            let path = Path::new(&path);
            let is_dir = path.is_dir();
            let (rotation_dir, prefix) = if is_dir {
                (path, DEFAULT_LOG_PREFIX)
            } else {
                let base_dir = path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .unwrap_or_else(|| Path::new("."));
                let prefix = path
                    .file_name()
                    .and_then(|file| file.to_str())
                    .unwrap_or(DEFAULT_LOG_PREFIX);
                (base_dir, prefix)
            };

            match RollingFileAppender::builder()
                .rotation(Rotation::DAILY)
                .filename_prefix(prefix)
                .max_log_files(7)
                .build(rotation_dir)
            {
                Ok(appender) => layer()
                    .with_writer(appender)
                    .with_target(true)
                    .with_ansi(false)
                    .boxed(),
                Err(err) => {
                    tracing::error!(
                        "failed to create rolling file appender at {}: {err}",
                        rotation_dir.display()
                    );
                    layer()
                        .with_writer(std::io::stdout)
                        .with_target(true)
                        .boxed()
                }
            }
        }
        _ => layer()
            .with_writer(std::io::stdout)
            .with_target(true)
            .boxed(),
    };

    fmt_layer.with_filter(filter).boxed()
}

pub fn init() {
    let log_path = env::var(LOG_PATH_ENV_VAR).ok();
    let fmt_layer = get_fmt_layer(log_path);
    registry().with(fmt_layer).init();
}

pub fn init_with_log_path(log_path: String) {
    let fmt_layer = get_fmt_layer(Some(log_path));
    registry().with(fmt_layer).init();
}
