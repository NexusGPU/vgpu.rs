//! CLI command implementations.

use std::fs;
use std::future::Future;
use std::os::unix::fs::PermissionsExt;
use std::pin::Pin;
use std::process::Command as StdCommand;

use anyhow::{Context, Result};
use utils::shared_memory::handle::SharedMemoryHandle;

use crate::app_builder::ApplicationBuilder;
use crate::config::{DaemonArgs, MountShmArgs, ShowShmArgs, ShowTuiWorkersArgs};
use crate::core::command::Command;
use crate::ui::tui::handlers::run_tui_monitor;

/// Daemon command implementation
pub struct DaemonCommand {
    args: DaemonArgs,
}

impl DaemonCommand {
    pub fn new(args: DaemonArgs) -> Self {
        Self { args }
    }
}

impl Command for DaemonCommand {
    fn execute(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            let _guard = crate::infrastructure::logging::init(self.args.gpu_metrics_file.clone());

            tracing::info!("Starting hypervisor daemon {}", &**utils::version::VERSION);

            let app = ApplicationBuilder::new(self.args.clone()).build().await?;

            app.run().await?;
            app.shutdown().await?;

            Ok(())
        })
    }

    fn name(&self) -> &'static str {
        "daemon"
    }

    fn description(&self) -> &'static str {
        "Run the hypervisor daemon"
    }
}

/// Mount shared memory command implementation
pub struct MountShmCommand {
    args: MountShmArgs,
}

impl MountShmCommand {
    pub fn new(args: MountShmArgs) -> Self {
        Self { args }
    }
}

impl Command for MountShmCommand {
    fn execute(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            utils::logging::init();

            tracing::info!("mount point: {:?}", self.args.mount_point);
            tracing::info!("size: {} MB", self.args.size_mb);

            // create mount point directory
            if !self.args.mount_point.exists() {
                tracing::info!("create mount point directory: {:?}", self.args.mount_point);
                fs::create_dir_all(&self.args.mount_point)
                    .context("create mount point directory failed")?;
            }

            // check if tmpfs is already mounted
            let mount_output = StdCommand::new("mount")
                .output()
                .context("execute mount command failed")?;

            let mount_info = String::from_utf8_lossy(&mount_output.stdout);
            let mount_point_str = self.args.mount_point.to_string_lossy();

            if mount_info.contains(&format!("on {mount_point_str} type tmpfs")) {
                tracing::info!("tmpfs is already mounted on {:?}", self.args.mount_point);
            } else {
                // mount tmpfs
                tracing::info!("mount tmpfs on {:?}", self.args.mount_point);
                let size_arg = format!("size={}M", self.args.size_mb);

                let mount_result = StdCommand::new("mount")
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
                fs::metadata(&self.args.mount_point).context("get mount point metadata failed")?;

            let mut permissions = metadata.permissions();
            permissions.set_mode(0o0777);

            fs::set_permissions(&self.args.mount_point, permissions)
                .context("set permissions failed")?;

            unsafe {
                libc::umask(old_umask);
            }

            Ok(())
        })
    }

    fn name(&self) -> &'static str {
        "mount-shm"
    }

    fn description(&self) -> &'static str {
        "Mount tmpfs for shared memory"
    }
}

/// Show shared memory command implementation
pub struct ShowShmCommand {
    args: ShowShmArgs,
}

impl ShowShmCommand {
    pub fn new(args: ShowShmArgs) -> Self {
        Self { args }
    }
}

impl Command for ShowShmCommand {
    fn execute(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            utils::logging::init();

            tracing::info!(
                "Attempting to open shared memory with identifier: {}",
                self.args.shm_identifier
            );

            let handle = SharedMemoryHandle::open(&self.args.shm_identifier)
                .context("Failed to open shared memory")?;

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

            Ok(())
        })
    }

    fn name(&self) -> &'static str {
        "show-shm"
    }

    fn description(&self) -> &'static str {
        "Show shared memory contents"
    }
}

/// Show TUI workers command implementation
pub struct ShowTuiWorkersCommand {
    args: ShowTuiWorkersArgs,
}

impl ShowTuiWorkersCommand {
    pub fn new(args: ShowTuiWorkersArgs) -> Self {
        Self { args }
    }
}

impl Command for ShowTuiWorkersCommand {
    fn execute(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            utils::logging::init_with_log_path(self.args.log_path.clone());

            tracing::info!(
                "Starting TUI worker monitor with pattern: {}",
                self.args.glob
            );

            // Note: Mock mode support can be added later if needed
            run_tui_monitor(format!("/dev/shm/{}", self.args.glob)).await
        })
    }

    fn name(&self) -> &'static str {
        "show-tui-workers"
    }

    fn description(&self) -> &'static str {
        "Show TUI workers monitor"
    }
}
