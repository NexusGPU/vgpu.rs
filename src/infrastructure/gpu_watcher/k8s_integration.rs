//! Kubernetes integration for GPU device state watching

use std::collections::{HashMap, HashSet};
use error_stack::{Report, ResultExt};
use kube::Api;
use tracing::{debug, error, info, warn};

use crate::infrastructure::k8s::types::KubernetesError;
use super::types::{GPU, GpuResourceStatus, KubeletDeviceState, PodDeviceEntry};

/// Extract device IDs from kubelet device state
pub fn extract_device_ids(
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
pub fn log_device_allocation_details(device_state: &KubeletDeviceState, device_id: &str) {
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
pub fn find_pod_for_device(
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
pub async fn patch_gpu_resource_for_added_device(
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
pub async fn patch_gpu_resource_for_removed_device(
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
pub fn build_resource_to_system_map(device_state: &KubeletDeviceState) -> HashMap<String, String> {
    let mut map = HashMap::new();
    
    for (system_name, _device_list) in &device_state.data.registered_devices {
        if system_name.starts_with("nvidia.com/") {
            map.insert(system_name.clone(), system_name.clone());
        }
    }
    
    debug!("Built resource to system map with {} entries", map.len());
    map
}