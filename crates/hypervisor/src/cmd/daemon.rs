use crate::config::DaemonArgs;
use crate::logging;
use crate::util::ApplicationBuilder;
use anyhow::Result;
use utils::version;

pub async fn run_daemon(daemon_args: DaemonArgs) -> Result<()> {
    let _guard = logging::init(daemon_args.gpu_metrics_file.clone());

    tracing::info!("Starting hypervisor daemon {}", &**version::VERSION);

    let app = ApplicationBuilder::new(daemon_args).build().await?;

    app.run().await?;
    app.shutdown().await?;

    Ok(())
}
