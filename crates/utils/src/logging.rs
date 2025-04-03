//! provides logging helpers

use tracing_subscriber::filter::{self};
use tracing_subscriber::fmt::layer;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry;

/// initiate the global tracing subscriber
pub fn init() {
    let env_filter = filter::EnvFilter::builder()
        .with_default_directive(filter::LevelFilter::INFO.into())
        .from_env_lossy();

    let fmt_layer = layer()
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_filter(env_filter);

    registry().with(fmt_layer).init();
}
