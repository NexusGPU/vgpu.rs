mod api;
mod app;
mod app_builder;
mod config;
mod gpu_allocation_watcher;
mod gpu_init;
mod gpu_observer;
mod host_pid_probe;
mod hypervisor;
mod k8s;
mod kube_client;
mod limiter_comm;
mod logging;
mod metrics;
mod pod_management;
mod process;
mod scheduler;
mod tui_workers;

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
        Commands::Daemon(daemon_args) => run_daemon(*daemon_args).await,
        Commands::MountShm(mount_shm_args) => run_mount_shm(mount_shm_args).await,
        Commands::ShowShm(show_shm_args) => run_show_shm(show_shm_args).await,
        Commands::ShowTuiWorkers(show_tui_workers_args) => {
            run_show_tui_workers(show_tui_workers_args).await
        }
    }
}

async fn run_show_shm(show_shm_args: crate::config::ShowShmArgs) -> Result<()> {
    use utils::shared_memory::handle::SharedMemoryHandle;
    utils::logging::init();

    tracing::info!(
        "Attempting to open shared memory with identifier: {}",
        show_shm_args.shm_identifier
    );

    let handle = SharedMemoryHandle::open(&show_shm_args.shm_identifier)
        .context("Failed to open shared memory")?;

    tracing::info!("Successfully opened shared memory handle");

    // Get the raw pointer for validation
    let ptr = handle.get_ptr();

    if ptr.is_null() {
        tracing::error!("Shared memory pointer is null!");
        return Err(anyhow::anyhow!("Shared memory pointer is null"));
    }

    tracing::info!("Shared memory pointer is valid: {:p}", ptr);

    // Get the state safely
    let state = handle.get_state();
    tracing::info!("Successfully accessed shared memory state");

    // Print basic information step by step
    let device_count = state.device_count();
    tracing::info!("Shared memory contains {} devices", device_count);

    let last_heartbeat = state.get_last_heartbeat();
    tracing::info!("Last heartbeat timestamp: {}", last_heartbeat);

    let device_uuids = state.get_device_uuids();
    tracing::info!("Device UUIDs: {:?}", device_uuids);

    // Try to check if the shared memory is healthy
    let is_healthy = state.is_healthy(60); // 60 seconds timeout
    tracing::info!("Shared memory health status (60s timeout): {}", is_healthy);

    // Print device details one by one
    for uuid in &device_uuids {
        if let Some(info) = state.with_device_by_uuid(uuid, |device| {
            format!("UUID: {}, Available cores: {}, Total cores: {}, Up limit: {}%, Memory limit: {} bytes, Pod memory used: {} bytes",
                uuid,
                device.get_available_cores(),
                device.get_total_cores(),
                device.get_up_limit(),
                device.get_mem_limit(),
                device.get_pod_memory_used()
            )
        }) {
            tracing::info!("Device info: {}", info);
        }
    }

    tracing::info!("Successfully completed shared memory inspection");
    Ok(())
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

    utils::logging::init();

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

    if mount_info.contains(&format!("on {mount_point_str} type tmpfs")) {
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
                &format!("rw,nosuid,nodev,{size_arg}"),
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

    let old_umask = unsafe { libc::umask(0) };

    // set directory permissions
    let metadata =
        fs::metadata(&mount_shm_args.mount_point).context("get mount point metadata failed")?;

    let mut permissions = metadata.permissions();
    permissions.set_mode(0o0777);

    fs::set_permissions(&mount_shm_args.mount_point, permissions)
        .context("set permissions failed")?;

    unsafe {
        libc::umask(old_umask);
    }

    Ok(())
}

async fn run_show_tui_workers(args: crate::config::ShowTuiWorkersArgs) -> Result<()> {
    utils::logging::init_with_log_path(args.log_path);

    tracing::info!("Starting TUI worker monitor with pattern: {}", args.glob);

    tui_workers::run_tui_monitor(format!("/dev/shm/{}", args.glob)).await
}
