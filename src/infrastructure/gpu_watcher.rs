//! GPU Device State Watcher Module
//! 
//! This module provides functionality to watch GPU device state changes
//! and synchronize them with Kubernetes custom resources.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::time::Duration;

use error_stack::{Report, ResultExt};
use k8s_openapi::ClusterResourceScope;
use kube::Api;
use notify::{Config, Event, RecommendedWatcher, Watcher};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::{fs, select, time::sleep};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::infrastructure::k8s::types::KubernetesError;
use crate::infrastructure::kube_client;

// ===== Data Types =====

/// Kubelet device state structure matching the JSON format
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct KubeletDeviceState {
    pub data: DeviceStateData,
    pub checksum: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct DeviceStateData {
    pub pod_device_entries: Option<Vec<PodDeviceEntry>>,
    pub registered_devices: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PodDeviceEntry {
    #[serde(rename = "PodUID")]
    pub pod_uid: String,
    pub container_name: String,
    pub resource_name: String,
    /// key is NUMA index, most case it is "-1", value is GPU ID
    #[serde(rename = "DeviceIDs")]
    pub device_ids: HashMap<String, Vec<String>>,
    #[serde(rename = "AllocResp")]
    pub alloc_resp: String,
}

/// GPU Custom Resource Definition for tensor-fusion.ai/v1
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct GpuResourceSpec {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dummy: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct GpuResourceStatus {
    pub used_by: Option<String>,
}

/// GPU Custom Resource with optional spec field
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
#[allow(clippy::upper_case_acronyms)]
pub struct GPU {
    #[serde(flatten)]
    pub metadata: kube::api::ObjectMeta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec: Option<GpuResourceSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<GpuResourceStatus>,
}

// Implement the Resource trait manually
impl kube::Resource for GPU {
    type DynamicType = ();
    type Scope = ClusterResourceScope;

    fn group(_dt: &()) -> std::borrow::Cow<'_, str> {
        "tensor-fusion.ai".into()
    }

    fn version(_dt: &()) -> std::borrow::Cow<'_, str> {
        "v1".into()
    }

    fn kind(_dt: &()) -> std::borrow::Cow<'_, str> {
        "GPU".into()
    }

    fn plural(_dt: &()) -> std::borrow::Cow<'_, str> {
        "gpus".into()
    }
}

// ===== Helper Functions =====

/// Extract device IDs from kubelet device state
fn extract_device_ids(
    device_state: &KubeletDeviceState,
    resource_to_system_map: &HashMap<String, String>,
) -> Result<HashSet<String>, Report<KubernetesError>> {
    let mut device_ids = HashSet::new();

    if let Some(pod_device_entries) = &device_state.data.pod_device_entries {
        for entry in pod_device_entries {
            // Extract GPU devices from resource allocation state
            if resource_to_system_map.contains_key(&entry.resource_name) {
                for device_list in entry.device_ids.values() {
                    for device_id in device_list {
                        device_ids.insert(device_id.to_lowercase());
                    }
                }
            }
        }
        debug!("Extracted {} unique device IDs", device_ids.len());
    }

    Ok(device_ids)
}

/// Log device allocation details for debugging
fn log_device_allocation_details(device_state: &KubeletDeviceState, device_id: &str) {
    if let Some(pod_device_entries) = &device_state.data.pod_device_entries {
        for entry in pod_device_entries {
            for device_list in entry.device_ids.values() {
                if device_list.contains(&device_id.to_string()) {
                    info!(
                        "Device allocation details - PodUID: {}, ContainerName: {}, ResourceName: {}, DeviceID: {}",
                        entry.pod_uid, entry.container_name, entry.resource_name, device_id
                    );
                }
            }
        }
    }
}

/// Find pod information for a specific device
fn find_pod_for_device(
    device_state: &KubeletDeviceState,
    device_id: &str,
    resource_to_system_map: &HashMap<String, String>,
) -> Option<String> {
    if let Some(pod_device_entries) = &device_state.data.pod_device_entries {
        for entry in pod_device_entries {
            if resource_to_system_map.contains_key(&entry.resource_name) {
                for device_list in entry.device_ids.values() {
                    if device_list
                        .iter()
                        .any(|id| id.to_lowercase() == device_id.to_lowercase())
                    {
                        return Some(format!("{}-{}", entry.pod_uid, entry.container_name));
                    }
                }
            }
        }
    }
    None
}

/// Patch GPU resource for added device
async fn patch_gpu_resource_for_added_device(
    gpu_api: &Api<GPU>,
    device_id: &str,
    device_state: &KubeletDeviceState,
    resource_to_system_map: &HashMap<String, String>,
) -> Result<(), Report<KubernetesError>> {
    let pod_info = find_pod_for_device(device_state, device_id, resource_to_system_map);
    
    let patch = serde_json::json!({
        "status": GpuResourceStatus {
            used_by: pod_info,
        }
    });

    match gpu_api
        .patch_status(device_id, &kube::api::PatchParams::default(), &kube::api::Patch::Merge(patch))
        .await
    {
        Ok(_) => {
            info!("Successfully patched GPU resource {} as allocated", device_id);
        }
        Err(kube::Error::Api(api_err)) if api_err.code == 404 => {
            warn!("GPU resource {} not found, skipping patch", device_id);
        }
        Err(e) => {
            return Err(Report::new(KubernetesError::UpdateFailed {
                message: format!("Failed to patch GPU resource {}: {}", device_id, e),
            }));
        }
    }
    
    Ok(())
}

/// Patch GPU resource for removed device
async fn patch_gpu_resource_for_removed_device(
    gpu_api: &Api<GPU>,
    device_id: &str,
) -> Result<(), Report<KubernetesError>> {
    let patch = serde_json::json!({
        "status": GpuResourceStatus {
            used_by: None,
        }
    });

    match gpu_api
        .patch_status(device_id, &kube::api::PatchParams::default(), &kube::api::Patch::Merge(patch))
        .await
    {
        Ok(_) => {
            info!("Successfully patched GPU resource {} as available", device_id);
        }
        Err(kube::Error::Api(api_err)) if api_err.code == 404 => {
            warn!("GPU resource {} not found, skipping patch", device_id);
        }
        Err(e) => {
            return Err(Report::new(KubernetesError::UpdateFailed {
                message: format!("Failed to patch GPU resource {}: {}", device_id, e),
            }));
        }
    }
    
    Ok(())
}

/// Build resource to system map from registered devices
fn build_resource_to_system_map(device_state: &KubeletDeviceState) -> HashMap<String, String> {
    let mut map = HashMap::new();
    
    for (system_name, _device_list) in &device_state.data.registered_devices {
        if system_name.starts_with("nvidia.com/") {
            map.insert(system_name.clone(), system_name.clone());
        }
    }
    
    debug!("Built resource to system map with {} entries", map.len());
    map
}

// ===== Main Watcher =====

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
        let resource_to_system_map = build_resource_to_system_map(&device_state);

        // Extract current device IDs
        let current_device_ids = extract_device_ids(&device_state, &resource_to_system_map)?;

        // Find added and removed devices
        let added_devices: HashSet<_> = current_device_ids.difference(previous_device_ids).collect();
        let removed_devices: HashSet<_> = previous_device_ids.difference(&current_device_ids).collect();

        // Process added devices
        for device_id in &added_devices {
            info!("Device added: {}", device_id);
            log_device_allocation_details(&device_state, device_id);

            if let Err(e) = patch_gpu_resource_for_added_device(
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

            if let Err(e) = patch_gpu_resource_for_removed_device(gpu_api, device_id).await {
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