
use std::path::PathBuf;
use std::collections::{HashMap, HashSet};
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use error_stack::{Report, ResultExt};
use crate::k8s::types::KubernetesError;
use std::time::Duration;
use crate::kube_client;
use kube::Api;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::time::sleep;
use schemars::JsonSchema;
use k8s_openapi::ClusterResourceScope;
use notify::{Config, Event, RecommendedWatcher, Watcher};
use tokio::sync::mpsc;
use std::sync::mpsc as std_mpsc;


pub struct GpuDeviceStateWatcher {
    kubelet_device_state_path: PathBuf,
}

impl GpuDeviceStateWatcher {
    pub fn new(kubelet_device_state_path: PathBuf) -> Self {
        Self { kubelet_device_state_path }
    }

    #[tracing::instrument(skip(self, cancellation_token))]
    pub(crate) async fn run(
        &self,
        cancellation_token: CancellationToken,
        kubeconfig: Option<PathBuf>,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Starting pod watcher");

        loop {
            select! {
                _ = cancellation_token.cancelled() => {
                    info!("Pod watcher shutdown requested");
                    break;
                }
                result = self.watch_and_patch_gpu_device_state(kubeconfig.clone()) => {
                    match result {
                        Ok(()) => {
                            warn!("Pod watch stream ended unexpectedly, restarting...");
                        }
                        Err(e) => {
                            error!("Pod watch failed: {e:?}");
                            // Wait before retrying
                            tokio::time::sleep(Duration::from_secs(5)).await;
                        }
                    }
                }
            }
        }

        Ok(())
    }


    #[tracing::instrument(skip(self))]
    async fn watch_and_patch_gpu_device_state(&self, kubeconfig: Option<PathBuf>) -> Result<(), Report<KubernetesError>> {
        info!("Starting GPU device state watcher");
        let client = kube_client::init_kube_client(kubeconfig).await?;
        let gpu_api: Api<GPU> = Api::all(client);
        
        let mut previous_device_ids = HashSet::new();
        let resource_to_system_map = Self::create_resource_system_map();
        
        info!("Starting GPU device state watcher for path: {:?}", self.kubelet_device_state_path);
        
        // Set up filesystem watcher
        let (fs_tx, mut fs_rx) = mpsc::channel(10);
        let watcher_result = self.setup_filesystem_watcher(fs_tx).await;
        
        match watcher_result {
            Ok(_watcher) => {
                // Keep watcher alive by holding it in scope
                info!("Filesystem watcher enabled for real-time updates");
                
                // Hybrid approach: filesystem events + periodic polling fallback
                let mut poll_interval = tokio::time::interval(Duration::from_secs(30)); // Reduced frequency since we have fs events
                poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                
                loop {
                    select! {
                        // Process filesystem events
                        Some(_event) = fs_rx.recv() => {
                            debug!("Filesystem event detected, processing device state");
                            if let Err(e) = self.read_and_process_device_state(&gpu_api, &mut previous_device_ids, &resource_to_system_map).await {
                                error!("Failed to process device state after filesystem event: {e:?}");
                            }
                        }
                        // Fallback polling every 30 seconds
                        _ = poll_interval.tick() => {
                            debug!("Periodic polling check");
                            if let Err(e) = self.read_and_process_device_state(&gpu_api, &mut previous_device_ids, &resource_to_system_map).await {
                                error!("Failed to process device state during periodic check: {e:?}");
                            }
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Failed to setup filesystem watcher, falling back to polling only: {e:?}");
                
                // Fallback to polling-only mode
                loop {
                    match self.read_and_process_device_state(&gpu_api, &mut previous_device_ids, &resource_to_system_map).await {
                        Ok(()) => {
                            debug!("Successfully processed device state");
                        }
                        Err(e) => {
                            error!("Failed to process device state: {e:?}");
                        }
                    }
                    
                    // Poll every 5 seconds in fallback mode
                    sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }
    
    /// Set up filesystem watcher for the kubelet device state file
    async fn setup_filesystem_watcher(
        &self,
        fs_tx: mpsc::Sender<Event>,
    ) -> Result<RecommendedWatcher, Report<KubernetesError>> {
        let (tx, rx) = std_mpsc::channel();
        
        // Create watcher
        let watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                match res {
                    Ok(event) => {
                        debug!("Filesystem event: {:?}", event);
                        // Send event to async channel (ignore errors if receiver is dropped)
                        let _ = tx.send(event);
                    }
                    Err(e) => {
                        error!("Filesystem watch error: {:?}", e);
                    }
                }
            },
            Config::default(),
        )
        .change_context(KubernetesError::ConnectionFailed { 
            message: "Failed to create filesystem watcher".to_string() 
        })?;
        
        // Spawn task to forward events from sync to async channel
        let fs_tx_clone = fs_tx.clone();
        tokio::spawn(async move {
            while let Ok(event) = rx.recv() {
                if fs_tx_clone.send(event).await.is_err() {
                    debug!("Filesystem event receiver dropped, stopping forwarder");
                    break;
                }
            }
        });
        
        Ok(watcher)
    }
    
    async fn read_and_process_device_state(
        &self,
        gpu_api: &Api<GPU>,
        previous_device_ids: &mut HashSet<String>,
        resource_to_system_map: &HashMap<String, String>,
    ) -> Result<(), Report<KubernetesError>> {
        // Read and parse the kubelet device state file
        let device_state = self.read_device_state_file().await?;
        
        // Extract current device IDs from PodDeviceEntries
        let current_device_ids = self.extract_device_ids(&device_state)?;
        
        // Find added and removed devices
        let added_devices: HashSet<_> = current_device_ids.difference(previous_device_ids).collect();
        let removed_devices: HashSet<_> = previous_device_ids.difference(&current_device_ids).collect();
        
        // Process added devices
        for device_id in &added_devices {
            info!("Device added: {}", device_id);
            self.log_device_allocation_details(&device_state, device_id);
            
            if let Err(e) = self.patch_gpu_resource_for_added_device(
                gpu_api, 
                device_id, 
                &device_state, 
                resource_to_system_map
            ).await {
                error!("Failed to patch GPU resource for added device {}: {e:?}", device_id);
            }
        }
        
        // Process removed devices
        for device_id in &removed_devices {
            info!("Device removed: {}", device_id);
            
            if let Err(e) = self.patch_gpu_resource_for_removed_device(gpu_api, device_id).await {
                error!("Failed to patch GPU resource for removed device {}: {e:?}", device_id);
            }
        }
        
        // Update previous state
        *previous_device_ids = current_device_ids;
        
        Ok(())
    }
    
    async fn read_device_state_file(&self) -> Result<KubeletDeviceState, Report<KubernetesError>> {
        let content = fs::read_to_string(&self.kubelet_device_state_path)
            .await
            .change_context(KubernetesError::WatchFailed { 
                message: format!("Failed to read device state file: {:?}", self.kubelet_device_state_path) 
            })?;
            
        serde_json::from_str(&content)
            .change_context(KubernetesError::AnnotationParseError { 
                message: "Failed to parse device state JSON".to_string() 
            })
    }
    
    fn extract_device_ids(&self, device_state: &KubeletDeviceState) -> Result<HashSet<String>, Report<KubernetesError>> {
        let mut device_ids = HashSet::new();
        
        for entry in &device_state.data.pod_device_entries {
            for device_list in entry.device_ids.values() {
                for device_id in device_list {
                    device_ids.insert(device_id.to_lowercase());
                }
            }
        }
        
        debug!("Extracted {} unique device IDs", device_ids.len());
        Ok(device_ids)
    }
    
    fn log_device_allocation_details(&self, device_state: &KubeletDeviceState, device_id: &str) {
        for entry in &device_state.data.pod_device_entries {
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
    
    async fn patch_gpu_resource_for_added_device(
        &self,
        gpu_api: &Api<GPU>,
        device_id: &str,
        device_state: &KubeletDeviceState,
        resource_to_system_map: &HashMap<String, String>,
    ) -> Result<(), Report<KubernetesError>> {
        // Find the resource name for this device
        let resource_name = self.find_resource_name_for_device(device_state, device_id)?;
        let used_by_system = resource_to_system_map.get(&resource_name)
            .unwrap_or(&"nvidia-device-plugin".to_string())
            .clone();
            
        info!("Patching GPU resource for device {} with usedBySystem: {}", device_id, used_by_system);
        
        self.patch_gpu_resource_with_retry(gpu_api, device_id, &used_by_system).await
    }
    
    async fn patch_gpu_resource_for_removed_device(
        &self,
        gpu_api: &Api<GPU>,
        device_id: &str,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Patching GPU resource for removed device {} with usedBySystem: tensor-fusion", device_id);
        
        self.patch_gpu_resource_with_retry(gpu_api, device_id, "tensor-fusion").await
    }
    
    async fn patch_gpu_resource_with_retry(
        &self,
        gpu_api: &Api<GPU>,
        device_id: &str,
        used_by_system: &str,
    ) -> Result<(), Report<KubernetesError>> {
        const MAX_RETRIES: u32 = 10;
        let mut retry_count = 0;
        
        while retry_count < MAX_RETRIES {
            match self.patch_gpu_resource(gpu_api, device_id, used_by_system).await {
                Ok(()) => {
                    info!("Successfully patched GPU resource for device: {}", device_id);
                    return Ok(());
                }
                Err(e) => {
                    retry_count += 1;
                    warn!("Failed to patch GPU resource (attempt {}/{}): {e:?}", retry_count, MAX_RETRIES);
                    
                    if retry_count < MAX_RETRIES {
                        let backoff_duration = Duration::from_millis(200 * (1 << retry_count));
                        sleep(backoff_duration).await;
                    }
                }
            }
        }
        
        Err(Report::new(KubernetesError::WatchFailed {
            message: format!("Failed to patch GPU resource after {} retries", MAX_RETRIES)
        }))
    }
    
    async fn patch_gpu_resource(
        &self,
        gpu_api: &Api<GPU>,
        device_id: &str,
        used_by_system: &str,
    ) -> Result<(), Report<KubernetesError>> {
        // Get current resource
        let mut current_resource = gpu_api.get(device_id).await
            .change_context(KubernetesError::WatchFailed {
                message: format!("Failed to get GPU resource: {}", device_id)
            })?;
        
        // Check if current status matches what we want to set
        if let Some(current_status) = &current_resource.status {
            if let Some(current_used_by) = &current_status.used_by {
                if current_used_by == used_by_system {
                    return Ok(()); // Already set to the desired value
                }
            }
        }

        // Create merge patch for status subresource using proper status structure
        current_resource.status.as_mut().map(|status| status.used_by = Some(used_by_system.to_string()));
       
        
        // Apply merge patch to status sub-resource
        gpu_api.patch_status(device_id, &kube::api::PatchParams::default(), &kube::api::Patch::Merge(&current_resource)).await
            .change_context(KubernetesError::WatchFailed {
                message: format!("Failed to patch GPU resource status: {}", device_id)
            })?;
            
        Ok(())
    }
    
    fn find_resource_name_for_device(
        &self,
        device_state: &KubeletDeviceState,
        device_id: &str,
    ) -> Result<String, Report<KubernetesError>> {
        for entry in &device_state.data.pod_device_entries {
            for device_list in entry.device_ids.values() {
                if device_list.iter().any(|d| d.to_lowercase() == device_id) {
                    return Ok(entry.resource_name.clone());
                }
            }
        }
        
        Err(Report::new(KubernetesError::AnnotationParseError {
            message: format!("Could not find resource name for device: {}", device_id)
        }))
    }
    
    fn create_resource_system_map() -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("nvidia.com/gpu".to_string(), "nvidia-device-plugin".to_string());
        map.insert("amd.com/gpu".to_string(), "amd-device-plugin".to_string());
        map.insert("intel.com/gpu".to_string(), "intel-device-plugin".to_string());
        map
    }
}

/// Kubelet device state structure matching the JSON format
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct KubeletDeviceState {
    data: DeviceStateData,
    checksum: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct DeviceStateData {
    pod_device_entries: Vec<PodDeviceEntry>,
    registered_devices: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct PodDeviceEntry {
    #[serde(rename = "PodUID")]
    pod_uid: String,
    container_name: String,
    resource_name: String,
    
    // key is NUMA index, most case it is "-1", value is GPU ID
    #[serde(rename = "DeviceIDs")]
    device_ids: HashMap<String, Vec<String>>,

    #[serde(rename = "AllocResp")]
    alloc_resp: String,
}

/// GPU Custom Resource Definition for tensor-fusion.ai/v1
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
struct GpuResourceSpec {
    #[serde(skip_serializing_if = "Option::is_none")]
    dummy: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
struct GpuResourceStatus {
    used_by: Option<String>,
}

/// GPU Custom Resource with optional spec field
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct GPU {
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
    
    fn meta(&self) -> &kube::api::ObjectMeta {
        &self.metadata
    }
    
    fn meta_mut(&mut self) -> &mut kube::api::ObjectMeta {
        &mut self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    use similar_asserts::assert_eq;

    fn create_test_device_state() -> KubeletDeviceState {
        let mut device_ids = HashMap::new();
        device_ids.insert("-1".to_string(), vec!["GPU-7d8429d5-531d-d6a6-6510-3b662081a75a".to_string()]);
        
        let pod_entry = PodDeviceEntry {
            pod_uid: "a7461dc1-023a-4bd5-a403-c738bb1d7db4".to_string(),
            container_name: "web".to_string(),
            resource_name: "nvidia.com/gpu".to_string(),
            device_ids,
            alloc_resp: "test-alloc-resp".to_string(),
        };
        
        let mut registered_devices = HashMap::new();
        registered_devices.insert(
            "nvidia.com/gpu".to_string(), 
            vec!["GPU-7d8429d5-531d-d6a6-6510-3b662081a75a".to_string()]
        );
        
        KubeletDeviceState {
            data: DeviceStateData {
                pod_device_entries: vec![pod_entry],
                registered_devices,
            },
            checksum: 2262205670,
        }
    }
    
    fn create_empty_device_state() -> KubeletDeviceState {
        KubeletDeviceState {
            data: DeviceStateData {
                pod_device_entries: vec![],
                registered_devices: HashMap::new(),
            },
            checksum: 0,
        }
    }
    
    async fn create_temp_device_state_file(device_state: &KubeletDeviceState) -> NamedTempFile {
        let mut temp_file = NamedTempFile::new().expect("should create temp file");
        let json_content = serde_json::to_string(device_state).expect("should serialize device state");
        use std::io::Write;
        temp_file.write_all(json_content.as_bytes()).expect("should write to temp file");
        temp_file.flush().expect("should flush temp file");
        temp_file
    }
    
    #[tokio::test]
    async fn test_read_device_state_file_success() {
        let device_state = create_test_device_state();
        let temp_file = create_temp_device_state_file(&device_state).await;
        
        let watcher = GpuDeviceStateWatcher::new(temp_file.path().to_path_buf());
        let result = watcher.read_device_state_file().await;
        
        assert!(result.is_ok(), "should successfully read device state file");
        let parsed_state = result.unwrap();
        assert_eq!(parsed_state.checksum, device_state.checksum, "checksum should match");
        assert_eq!(parsed_state.data.pod_device_entries.len(), 1, "should have one pod device entry");
    }
    
    #[tokio::test]
    async fn test_read_device_state_file_not_found() {
        let watcher = GpuDeviceStateWatcher::new(PathBuf::from("/nonexistent/path"));
        let result = watcher.read_device_state_file().await;
        
        assert!(result.is_err(), "should fail when file does not exist");
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Failed to read device state file"), "error should mention file read failure");
    }
    
    #[tokio::test]
    async fn test_read_device_state_file_invalid_json() {
        let mut temp_file = NamedTempFile::new().expect("should create temp file");
        use std::io::Write;
        temp_file.write_all(b"invalid json content").expect("should write to temp file");
        temp_file.flush().expect("should flush temp file");
        
        let watcher = GpuDeviceStateWatcher::new(temp_file.path().to_path_buf());
        let result = watcher.read_device_state_file().await;
        
        assert!(result.is_err(), "should fail when JSON is invalid");
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Failed to parse device state JSON"), "error should mention JSON parse failure");
    }
    
    #[test]
    fn test_extract_device_ids_with_devices() {
        let watcher = GpuDeviceStateWatcher::new(PathBuf::from("/test"));
        let device_state = create_test_device_state();
        
        let result = watcher.extract_device_ids(&device_state);
        
        assert!(result.is_ok(), "should successfully extract device IDs");
        let device_ids = result.unwrap();
        assert_eq!(device_ids.len(), 1, "should extract one device ID");
        assert!(device_ids.contains("gpu-7d8429d5-531d-d6a6-6510-3b662081a75a"), "should contain expected device ID");
    }
    
    #[test]
    fn test_extract_device_ids_empty() {
        let watcher = GpuDeviceStateWatcher::new(PathBuf::from("/test"));
        let device_state = create_empty_device_state();
        
        let result = watcher.extract_device_ids(&device_state);
        
        assert!(result.is_ok(), "should successfully extract empty device IDs");
        let device_ids = result.unwrap();
        assert_eq!(device_ids.len(), 0, "should extract no device IDs");
    }
    
    #[test]
    fn test_extract_device_ids_multiple_devices() {
        let watcher = GpuDeviceStateWatcher::new(PathBuf::from("/test"));
        
        let mut device_ids_1 = HashMap::new();
        device_ids_1.insert("-1".to_string(), vec!["GPU-1".to_string(), "GPU-2".to_string()]);
        
        let mut device_ids_2 = HashMap::new();
        device_ids_2.insert("-1".to_string(), vec!["GPU-3".to_string()]);
        
        let pod_entry_1 = PodDeviceEntry {
            pod_uid: "pod-1".to_string(),
            container_name: "container-1".to_string(),
            resource_name: "nvidia.com/gpu".to_string(),
            device_ids: device_ids_1,
            alloc_resp: "alloc-1".to_string(),
        };
        
        let pod_entry_2 = PodDeviceEntry {
            pod_uid: "pod-2".to_string(),
            container_name: "container-2".to_string(),
            resource_name: "nvidia.com/gpu".to_string(),
            device_ids: device_ids_2,
            alloc_resp: "alloc-2".to_string(),
        };
        
        let device_state = KubeletDeviceState {
            data: DeviceStateData {
                pod_device_entries: vec![pod_entry_1, pod_entry_2],
                registered_devices: HashMap::new(),
            },
            checksum: 12345,
        };
        
        let result = watcher.extract_device_ids(&device_state);
        
        assert!(result.is_ok(), "should successfully extract multiple device IDs");
        let device_ids = result.unwrap();
        assert_eq!(device_ids.len(), 3, "should extract three unique device IDs");
        assert!(device_ids.contains("gpu-1"), "should contain GPU-1");
        assert!(device_ids.contains("gpu-2"), "should contain GPU-2");
        assert!(device_ids.contains("gpu-3"), "should contain GPU-3");
    }
    
    #[test]
    fn test_find_resource_name_for_device_success() {
        let watcher = GpuDeviceStateWatcher::new(PathBuf::from("/test"));
        let device_state = create_test_device_state();
        
        let result = watcher.find_resource_name_for_device(&device_state, "gpu-7d8429d5-531d-d6a6-6510-3b662081a75a");
        
        assert!(result.is_ok(), "should successfully find resource name");
        let resource_name = result.unwrap();
        assert_eq!(resource_name, "nvidia.com/gpu", "should return correct resource name");
    }
    
    #[test]
    fn test_find_resource_name_for_device_not_found() {
        let watcher = GpuDeviceStateWatcher::new(PathBuf::from("/test"));
        let device_state = create_test_device_state();
        
        let result = watcher.find_resource_name_for_device(&device_state, "nonexistent-device");
        
        assert!(result.is_err(), "should fail when device not found");
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Could not find resource name for device"), "error should mention device not found");
    }
    
    #[test]
    fn test_create_resource_system_map() {
        let map = GpuDeviceStateWatcher::create_resource_system_map();
        
        assert_eq!(map.len(), 3, "should contain three resource mappings");
        assert_eq!(map.get("nvidia.com/gpu"), Some(&"nvidia-device-plugin".to_string()), "should map nvidia.com/gpu correctly");
        assert_eq!(map.get("amd.com/gpu"), Some(&"amd-device-plugin".to_string()), "should map amd.com/gpu correctly");
        assert_eq!(map.get("intel.com/gpu"), Some(&"intel-device-plugin".to_string()), "should map intel.com/gpu correctly");
    }
    
    #[test]
    fn test_log_device_allocation_details() {
        let watcher = GpuDeviceStateWatcher::new(PathBuf::from("/test"));
        let device_state = create_test_device_state();
        
        // This test mainly ensures the function doesn't panic
        // In a real scenario, we would capture log output to verify the content
        watcher.log_device_allocation_details(&device_state, "GPU-7d8429d5-531d-d6a6-6510-3b662081a75a");
    }
    
    #[test]
    fn test_kubelet_device_state_serialization() {
        let device_state = create_test_device_state();
        
        // Test serialization
        let json_str = serde_json::to_string(&device_state).expect("should serialize to JSON");
        assert!(json_str.contains("PodDeviceEntries"), "JSON should contain PodDeviceEntries");
        assert!(json_str.contains("RegisteredDevices"), "JSON should contain RegisteredDevices");
        
        // Test deserialization
        let deserialized: KubeletDeviceState = serde_json::from_str(&json_str).expect("should deserialize from JSON");
        assert_eq!(deserialized.checksum, device_state.checksum, "checksum should match after round-trip");
        assert_eq!(deserialized.data.pod_device_entries.len(), device_state.data.pod_device_entries.len(), "pod entries count should match");
    }
    
    #[test]
    fn test_gpu_resource_status_default() {
        let status = GpuResourceStatus::default();
        assert_eq!(status.used_by, None, "default status should have None for used_by");
    }
    
    #[test]
    fn test_gpu_resource_serialization() {
        let mut status = GpuResourceStatus::default();
        status.used_by = Some("nvidia-device-plugin".to_string());
        
        let json_str = serde_json::to_string(&status).expect("should serialize GPU resource status");
        assert!(json_str.contains("usedBy"), "JSON should use camelCase for used_by field");
        
        let deserialized: GpuResourceStatus = serde_json::from_str(&json_str).expect("should deserialize GPU resource status");
        assert_eq!(deserialized.used_by, Some("nvidia-device-plugin".to_string()), "used_by should match after round-trip");
    }
    
    #[test]
    fn test_gpu_resource_without_spec_deserialization() {
        // Test that we can deserialize a GPU resource without a spec field
        let json_without_spec = r#"{
            "apiVersion": "tensor-fusion.ai/v1",
            "kind": "GPU",
            "metadata": {
                "name": "gpu-test",
                "namespace": "default"
            },
            "status": {
                "usedBy": "nvidia-device-plugin"
            }
        }"#;
        
        let result: Result<GPU, _> = serde_json::from_str(json_without_spec);
        assert!(result.is_ok(), "should successfully deserialize GPU resource without spec field");
        
        let gpu = result.unwrap();
        assert!(gpu.spec.is_none(), "spec should be None when not present in JSON");
        assert!(gpu.status.is_some(), "status should be present");
        assert_eq!(gpu.status.unwrap().used_by, Some("nvidia-device-plugin".to_string()), "used_by should match");
    }
    
    #[test]
    fn test_gpu_resource_with_spec_deserialization() {
        // Test that we can still deserialize a GPU resource with a spec field
        let json_with_spec = r#"{
            "apiVersion": "tensor-fusion.ai/v1",
            "kind": "GPU",
            "metadata": {
                "name": "gpu-test",
                "namespace": "default"
            },
            "spec": {
                "dummy": "test-value"
            },
            "status": {
                "usedBy": "nvidia-device-plugin"
            }
        }"#;
        
        let result: Result<GPU, _> = serde_json::from_str(json_with_spec);
        assert!(result.is_ok(), "should successfully deserialize GPU resource with spec field");
        
        let gpu = result.unwrap();
        assert!(gpu.spec.is_some(), "spec should be present when included in JSON");
        assert_eq!(gpu.spec.unwrap().dummy, Some("test-value".to_string()), "dummy field should match");
        assert!(gpu.status.is_some(), "status should be present");
        assert_eq!(gpu.status.unwrap().used_by, Some("nvidia-device-plugin".to_string()), "used_by should match");
    }
    
    #[test]
    fn test_device_state_with_complex_device_ids() {
        let watcher = GpuDeviceStateWatcher::new(PathBuf::from("/test"));
        
        // Create a more complex device ID structure
        let mut device_ids = HashMap::new();
        device_ids.insert("-1".to_string(), vec!["GPU-1".to_string()]);
        device_ids.insert("0".to_string(), vec!["GPU-2".to_string(), "GPU-3".to_string()]);
        
        let pod_entry = PodDeviceEntry {
            pod_uid: "complex-pod".to_string(),
            container_name: "complex-container".to_string(),
            resource_name: "nvidia.com/gpu".to_string(),
            device_ids,
            alloc_resp: "complex-alloc".to_string(),
        };
        
        let device_state = KubeletDeviceState {
            data: DeviceStateData {
                pod_device_entries: vec![pod_entry],
                registered_devices: HashMap::new(),
            },
            checksum: 54321,
        };
        
        let result = watcher.extract_device_ids(&device_state);
        assert!(result.is_ok(), "should handle complex device ID structure");
        
        let device_ids = result.unwrap();
        assert_eq!(device_ids.len(), 3, "should extract all three device IDs");
        assert!(device_ids.contains("gpu-1"), "should contain GPU-1");
        assert!(device_ids.contains("gpu-2"), "should contain GPU-2");
        assert!(device_ids.contains("gpu-3"), "should contain GPU-3");
    }
}
