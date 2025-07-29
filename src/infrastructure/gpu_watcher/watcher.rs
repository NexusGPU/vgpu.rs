//! Main GPU device state watcher implementation

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::time::Duration;

use error_stack::{Report, ResultExt};
use kube::Api;
use notify::{Config, Event, RecommendedWatcher, Watcher};
use tokio::{fs, select, time::sleep};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::infrastructure::k8s::types::KubernetesError;
use crate::infrastructure::kube_client;
use super::types::{GPU, KubeletDeviceState};
use super::k8s_integration;

/// GPU device state watcher that monitors kubelet device state changes
/// and synchronizes with Kubernetes GPU custom resources
pub struct GpuDeviceStateWatcher {
    kubelet_device_state_path: PathBuf,
}

impl GpuDeviceStateWatcher {
    /// Create a new GPU device state watcher
    pub fn new(kubelet_device_state_path: PathBuf) -> Self {
        Self {
            kubelet_device_state_path,
        }
    }

    /// Run the watcher with cancellation support
    #[tracing::instrument(skip(self, cancellation_token))]
    pub async fn run(
        &self,
        cancellation_token: CancellationToken,
        kubeconfig: Option<PathBuf>,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Starting GPU allocation watcher");

        loop {
            select! {
                _ = cancellation_token.cancelled() => {
                    info!("GPU allocation watcher shutdown requested");
                    break;
                }
                result = self.watch_and_patch_gpu_device_state(kubeconfig.clone()) => {
                    match result {
                        Ok(()) => {
                            warn!("GPU allocation watch stream ended unexpectedly, restarting...");
                        }
                        Err(e) => {
                            error!("GPU allocation watch failed: {e:?}");
                            // Wait before retrying
                            sleep(Duration::from_secs(5)).await;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Main watching logic
    #[tracing::instrument(skip(self))]
    async fn watch_and_patch_gpu_device_state(
        &self,
        kubeconfig: Option<PathBuf>,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Starting GPU device state watcher");
        let client = kube_client::init_kube_client(kubeconfig).await?;
        let gpu_api: Api<GPU> = Api::all(client);

        // Set up file watcher
        let (tx, rx) = std_mpsc::channel();
        let mut watcher = RecommendedWatcher::new(tx, Config::default())
            .change_context(KubernetesError::WatchFailed {
                message: "Failed to create file watcher".to_string(),
            })?;

        watcher
            .watch(&self.kubelet_device_state_path, notify::RecursiveMode::NonRecursive)
            .change_context(KubernetesError::WatchFailed {
                message: format!(
                    "Failed to watch file: {:?}",
                    self.kubelet_device_state_path
                ),
            })?;

        info!(
            "Watching device state file: {:?}",
            self.kubelet_device_state_path
        );

        // Initial processing
        let mut previous_device_ids = HashSet::new();
        if let Err(e) = self.process_device_state_change(&gpu_api, &mut previous_device_ids).await {
            warn!("Initial device state processing failed: {e:?}");
        }

        // Watch for file changes
        for res in rx {
            match res {
                Ok(event) => {
                    debug!("File event: {:?}", event);
                    if self.should_process_event(&event) {
                        if let Err(e) = self
                            .process_device_state_change(&gpu_api, &mut previous_device_ids)
                            .await
                        {
                            error!("Failed to process device state change: {e:?}");
                        }
                    }
                }
                Err(e) => {
                    error!("File watch error: {e:?}");
                    return Err(Report::new(KubernetesError::WatchFailed {
                        message: format!("File watch error: {e:?}"),
                    }));
                }
            }
        }

        Ok(())
    }

    /// Check if file event should trigger processing
    fn should_process_event(&self, event: &Event) -> bool {
        matches!(
            event.kind,
            notify::EventKind::Modify(_) | notify::EventKind::Create(_)
        )
    }

    /// Process device state changes and update Kubernetes resources
    async fn process_device_state_change(
        &self,
        gpu_api: &Api<GPU>,
        previous_device_ids: &mut HashSet<String>,
    ) -> Result<(), Report<KubernetesError>> {
        debug!("Processing device state change");

        let device_state = self.read_device_state_file().await?;
        let resource_to_system_map = k8s_integration::build_resource_to_system_map(&device_state);

        // Extract current device IDs
        let current_device_ids = k8s_integration::extract_device_ids(&device_state, &resource_to_system_map)?;

        // Find added and removed devices
        let added_devices: HashSet<_> = current_device_ids.difference(previous_device_ids).collect();
        let removed_devices: HashSet<_> = previous_device_ids.difference(&current_device_ids).collect();

        // Process added devices
        for device_id in &added_devices {
            info!("Device added: {}", device_id);
            k8s_integration::log_device_allocation_details(&device_state, device_id);

            if let Err(e) = k8s_integration::patch_gpu_resource_for_added_device(
                gpu_api,
                device_id,
                &device_state,
                &resource_to_system_map,
            )
            .await
            {
                error!("Failed to patch GPU resource for added device {}: {e:?}", device_id);
            }
        }

        // Process removed devices
        for device_id in &removed_devices {
            info!("Device removed: {}", device_id);

            if let Err(e) = k8s_integration::patch_gpu_resource_for_removed_device(gpu_api, device_id).await {
                error!("Failed to patch GPU resource for removed device {}: {e:?}", device_id);
            }
        }

        // Update previous state
        *previous_device_ids = current_device_ids;

        Ok(())
    }

    /// Read and parse device state file
    async fn read_device_state_file(&self) -> Result<KubeletDeviceState, Report<KubernetesError>> {
        let content = fs::read_to_string(&self.kubelet_device_state_path)
            .await
            .change_context(KubernetesError::WatchFailed {
                message: format!(
                    "Failed to read device state file: {:?}",
                    self.kubelet_device_state_path
                ),
            })?;

        serde_json::from_str(&content).change_context(KubernetesError::AnnotationParseError {
            message: "Failed to parse device state JSON".to_string(),
        })
    }
}