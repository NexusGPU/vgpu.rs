use anyhow::{Context, Result};
use clap::Parser;
use std::time::Duration;
use utils::version;

use hypervisor::app::ApplicationBuilder;
use hypervisor::config::{Cli, Commands};
use hypervisor::{logging, tui};

/// Sets up global panic hooks.
fn setup_panic() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        crossterm::terminal::disable_raw_mode().unwrap();
        crossterm::execute!(std::io::stderr(), crossterm::terminal::LeaveAlternateScreen).unwrap();
        original_hook(panic_info);
    }));
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_panic();

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

async fn run_daemon(daemon_args: hypervisor::config::DaemonArgs) -> Result<()> {
    let _guard = logging::init(daemon_args.gpu_metrics_file.clone());

    tracing::info!("Starting hypervisor daemon {}", &**version::VERSION);

    let app = ApplicationBuilder::new(daemon_args).build().await?;

    app.run().await?;
    app.shutdown().await?;

    Ok(())
}

async fn run_show_shm(show_shm_args: hypervisor::config::ShowShmArgs) -> Result<()> {
    use utils::shared_memory::handle::SharedMemoryHandle;
    use utils::shared_memory::PodIdentifier;
    utils::logging::init();

    // Create PodIdentifier and construct shared memory path
    let pod_identifier = PodIdentifier::new(&show_shm_args.namespace, &show_shm_args.pod_name);
    let shm_path = pod_identifier.to_path(&show_shm_args.shm_base_path);

    tracing::info!(
        "Attempting to open shared memory for pod {}/{} at path: {}",
        show_shm_args.namespace,
        show_shm_args.pod_name,
        shm_path.display()
    );

    let handle = SharedMemoryHandle::open(&shm_path)?;

    // Get the raw pointer for validation
    let ptr = handle.get_ptr();

    if ptr.is_null() {
        tracing::error!("Shared memory pointer is null!");
        return Err(anyhow::anyhow!("Shared memory pointer is null"));
    }

    // Get the state safely
    let state = handle.get_state();

    // Print basic information step by step
    let device_count = state.device_count();
    tracing::info!("Shared memory contains {} devices", device_count);

    let last_heartbeat = state.get_last_heartbeat();
    tracing::info!("Last heartbeat timestamp: {}", last_heartbeat);

    // Try to check if the shared memory is healthy
    let is_healthy = state.is_healthy(Duration::from_secs(2)); // 60 seconds timeout
    tracing::info!("Shared memory health status (2s timeout): {}", is_healthy);

    // Print version information
    let version = state.get_version();
    tracing::info!("Shared memory state version: v{}", version);

    // Print device details one by one
    state.for_each_active_device(|_, device| {
        let info = format!("uuid: {}, Available cores: {}, Total cores: {}, Up limit: {}%, Memory limit: {} bytes, Pod memory used: {} bytes",
            device.get_uuid(),
            device.device_info.get_available_cores(),
            device.device_info.get_total_cores(),
            device.device_info.get_up_limit(),
            device.device_info.get_mem_limit(),
            device.device_info.get_pod_memory_used()
        );

        tracing::info!("Device info: {}", info);
    });

    // Print additional state information
    let (heartbeat, pids, state_version) = state.get_detailed_state_info();
    tracing::info!(
        "Detailed state - Heartbeat: {}, PIDs count: {}, State version: {}",
        heartbeat,
        pids.len(),
        state_version
    );

    if !pids.is_empty() {
        tracing::info!("Active PIDs: {:?}", pids);
    }

    // Print individual device information using the new API
    for i in 0..device_count {
        if let Some((
            uuid,
            available_cores,
            total_cores,
            mem_limit,
            pod_memory_used,
            up_limit,
            is_active,
        )) = state.get_device_info(i)
        {
            tracing::info!(
                "Device {}: UUID={}, Available={}, Total={}, MemLimit={}, MemUsed={}, UpLimit={}%, Active={}",
                i, uuid, available_cores, total_cores, mem_limit, pod_memory_used, up_limit, is_active
            );
        }
    }

    Ok(())
}

async fn run_mount_shm(mount_shm_args: hypervisor::config::MountShmArgs) -> Result<()> {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    utils::logging::init();

    tracing::info!("mount point: {:?}", mount_shm_args.mount_point);
    tracing::info!("size: {} MB", mount_shm_args.size_mb);

    // Clean up shared memory files if prefix is provided
    if let Some(ref cleanup_prefix) = mount_shm_args.cleanup_prefix {
        tracing::info!(
            "Cleaning up ALL shared memory files with prefix: {}",
            cleanup_prefix
        );

        let cleanup_result = cleanup_all_shared_memory_files(cleanup_prefix);

        match cleanup_result {
            Ok(cleaned_files) => {
                if !cleaned_files.is_empty() {
                    tracing::info!(
                        "Cleaned up {} shared memory files: {:?}",
                        cleaned_files.len(),
                        cleaned_files
                    );
                } else {
                    tracing::info!(
                        "No shared memory files found with prefix: {}",
                        cleanup_prefix
                    );
                }
            }
            Err(e) => {
                tracing::warn!("Failed to cleanup shared memory files: {}", e);
            }
        }
    }

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

/// Cleans up ALL shared memory files in /dev/shm with the specified prefix pattern
/// This function directly removes files without orphan detection
fn cleanup_all_shared_memory_files(prefix_pattern: &str) -> Result<Vec<String>> {
    use std::fs;

    let mut cleaned_files = Vec::new();

    // Use glob to find matching files in /dev/shm
    let pattern = format!("/dev/shm/{prefix_pattern}");
    let paths = glob::glob(&pattern)
        .with_context(|| format!("Failed to compile glob pattern: {pattern}"))?;

    for path_result in paths {
        let file_path = path_result
            .with_context(|| format!("Failed to read glob path for pattern: {pattern}"))?;

        if !file_path.is_file() {
            continue;
        }

        // Extract filename for logging
        let filename = file_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("<unknown>");

        // Directly remove the file without any orphan checks
        match fs::remove_file(&file_path) {
            Ok(_) => {
                cleaned_files.push(filename.to_string());
                tracing::info!("Removed shared memory file: {}", file_path.display());
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to remove shared memory file {}: {}",
                    file_path.display(),
                    e
                );
            }
        }
    }

    Ok(cleaned_files)
}

async fn run_show_tui_workers(args: hypervisor::config::ShowTuiWorkersArgs) -> Result<()> {
    utils::logging::init_with_log_path(args.log_path);

    tracing::info!("Starting TUI worker monitor with pattern: {}", args.glob);

    if args.mock {
        tracing::info!("Starting TUI in mock mode for local debugging");
        tui::handlers::run_tui_monitor_mock().await
    } else {
        tui::handlers::run_tui_monitor(format!("/dev/shm/{}", args.glob)).await
    }
}
