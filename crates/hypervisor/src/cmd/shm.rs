use crate::config::{MountShmArgs, ShowShmArgs};
use anyhow::{Context, Result};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;
use std::time::Duration;
use utils::shared_memory::handle::SharedMemoryHandle;
use utils::shared_memory::PodIdentifier;

pub async fn run_show_shm(show_shm_args: ShowShmArgs) -> Result<()> {
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
    let is_healthy = state.is_healthy(Duration::from_secs(2)); // 2 seconds timeout
    tracing::info!("Shared memory health status (2s timeout): {}", is_healthy);

    // Print version information
    let version = state.get_version();
    tracing::info!("Shared memory state version: v{}", version);

    // Print device details depending on version
    match state.get_version() {
        1 => {
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
                    if is_active {
                        tracing::info!(
                            "Device {}: UUID={}, Available={}, Total={}, MemLimit={}, MemUsed={}, UpLimit={}%, Active={}",
                            i, uuid, available_cores, total_cores, mem_limit, pod_memory_used, up_limit, is_active
                        );
                    }
                }
            }
        }
        2 => {
            for (i, device) in state.iter_active_devices() {
                tracing::info!(
                    "Device {}: UUID={}, TotalCores={}, MemLimit={}, MemUsed={}, UpLimit={}%, ERL(avg_cost={:.3}, tokens={:.3}/{:.3}, refill={:.3}/s, last_update={:.0})",
                    i,
                    device.get_uuid(),
                    device.device_info.get_total_cores(),
                    device.device_info.get_mem_limit(),
                    device.device_info.get_pod_memory_used(),
                    device.device_info.get_up_limit(),
                    device.device_info.get_erl_token_refill_rate(),
                    device.device_info.get_erl_current_tokens(),
                    device.device_info.get_erl_token_capacity(),
                    device.device_info.get_erl_token_refill_rate(),
                    device.device_info.get_erl_last_token_update(),
                );
            }
        }
        other => {
            tracing::warn!(
                version = other,
                "Unknown shared memory version; skipping device details"
            );
        }
    }

    // Print additional state information
    let heartbeat = state.get_last_heartbeat();
    let pids = state.get_all_pids();
    let state_version = state.get_version();
    tracing::info!(
        "Detailed state - Heartbeat: {}, PIDs count: {}, State version: {}",
        heartbeat,
        pids.len(),
        state_version
    );

    if !pids.is_empty() {
        tracing::info!("Active PIDs: {:?}", pids);
    }

    // Detailed device information printed above per version

    Ok(())
}

pub async fn run_mount_shm(mount_shm_args: MountShmArgs) -> Result<()> {
    utils::logging::init();

    tracing::info!("mount point: {:?}", mount_shm_args.mount_point);
    tracing::info!("size: {} MB", mount_shm_args.size_mb);

    tracing::info!(
        "Cleaning up ALL shared memory files in: {:?}",
        mount_shm_args.mount_point
    );

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
