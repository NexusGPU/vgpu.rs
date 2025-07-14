mod api;
mod app;
mod app_builder;
mod config;
mod gpu_init;
mod gpu_observer;
mod host_pid_probe;
mod hypervisor;
mod k8s;
mod limiter_comm;
mod limiter_coordinator;
mod logging;
mod metrics;
mod process;
mod scheduler;
mod worker_manager;
mod worker_registration;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use utils::version;

use crate::app_builder::ApplicationBuilder;
use crate::config::Cli;
use crate::config::Commands;

/// Sets up global panic hooks.
fn setup_global_hooks() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        default_hook(panic_info);
        tracing::error!("Thread panicked: {}", panic_info);
    }));
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_global_hooks();

    let cli = Cli::parse();

    match cli.command {
        Commands::Daemon(daemon_args) => run_daemon(daemon_args).await,
        Commands::MountShm(mount_shm_args) => run_mount_shm(mount_shm_args).await,
    }
}

async fn run_daemon(daemon_args: crate::config::DaemonArgs) -> Result<()> {
    let _guard = logging::init(daemon_args.gpu_metrics_file.clone());

    tracing::info!("Starting hypervisor daemon {}", &**version::VERSION);

    let app = ApplicationBuilder::new(daemon_args).build().await?;

    app.run().await?;
    app.shutdown().await?;

    Ok(())
}

async fn run_mount_shm(mount_shm_args: crate::config::MountShmArgs) -> Result<()> {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    tracing::info!("mount point: {:?}", mount_shm_args.mount_point);
    tracing::info!("size: {} MB", mount_shm_args.size_mb);

    // create mount point directory
    if !mount_shm_args.mount_point.exists() {
        tracing::info!(
            "create mount point directory: {:?}",
            mount_shm_args.mount_point
        );
        fs::create_dir_all(&mount_shm_args.mount_point)
            .context("create mount point directory failed")?;
    }

    // check if tmpfs is already mounted
    let mount_output = Command::new("mount")
        .output()
        .context("execute mount command failed")?;

    let mount_info = String::from_utf8_lossy(&mount_output.stdout);
    let mount_point_str = mount_shm_args.mount_point.to_string_lossy();

    if mount_info.contains(&format!("on {} type tmpfs", mount_point_str)) {
        tracing::info!(
            "tmpfs is already mounted on {:?}",
            mount_shm_args.mount_point
        );
    } else {
        // mount tmpfs
        tracing::info!("mount tmpfs on {:?}", mount_shm_args.mount_point);
        let size_arg = format!("size={}M", mount_shm_args.size_mb);

        let mount_result = Command::new("mount")
            .args([
                "-t",
                "tmpfs",
                "-o",
                &format!("rw,nosuid,nodev,{}", size_arg),
                "tmpfs",
                mount_point_str.as_ref(),
            ])
            .status()
            .context("execute mount command failed")?;

        if !mount_result.success() {
            return Err(anyhow::anyhow!("mount tmpfs failed"));
        }

        tracing::info!("mount tmpfs successfully");
    }

    // set directory permissions
    let metadata =
        fs::metadata(&mount_shm_args.mount_point).context("get mount point metadata failed")?;

    let mut permissions = metadata.permissions();
    permissions.set_mode(0755);

    fs::set_permissions(&mount_shm_args.mount_point, permissions)
        .context("set permissions failed")?;
    Ok(())
}
