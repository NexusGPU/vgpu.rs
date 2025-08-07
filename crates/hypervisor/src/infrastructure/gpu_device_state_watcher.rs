use error_stack::Report;
use error_stack::ResultExt;
use k8s_openapi::ClusterResourceScope;
use kube::Api;
use notify::Config;
use notify::Event;
use notify::RecommendedWatcher;
use notify::Watcher;
use rand::Rng;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::HashSet;
use std::io;
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::time::Duration;
use tokio::fs;
use tokio::select;
use tokio::sync::mpsc;
use tokio::time::interval;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::k8s::types::KubernetesError;
use crate::kube_client;
use kube::api::{ObjectMeta, Patch};

// Include generated protobuf code
pub mod pod_resources {
    tonic::include_proto!("v1");
}

pub struct GpuDeviceStateWatcher {
    kubelet_device_state_path: PathBuf,
    kubelet_socket_path: PathBuf,
}

impl GpuDeviceStateWatcher {
    pub fn new<P1: Into<PathBuf>, P2: Into<PathBuf>>(
        kubelet_device_state_path: P1,
        kubelet_socket_path: P2,
    ) -> Self {
        Self {
            kubelet_device_state_path: kubelet_device_state_path.into(),
            kubelet_socket_path: kubelet_socket_path.into(),
        }
    }

    /// Create a duration with jitter to avoid thundering herd problems
    fn duration_with_jitter(base_duration: Duration, jitter_percent: f64) -> Duration {
        let mut rng = rand::rng();
        let jitter_range = base_duration.as_secs_f64() * jitter_percent;

        let jitter_offset = rng.random_range(-jitter_range..=jitter_range);
        let final_duration = base_duration.as_secs_f64() + jitter_offset;

        Duration::from_secs_f64(final_duration)
    }

    #[tracing::instrument(skip(self, cancellation_token))]
    pub(crate) async fn run(
        &self,
        cancellation_token: CancellationToken,
        kubeconfig: Option<PathBuf>,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Starting gpu allocation watcher");

        let mut previous_device_ids = HashSet::new();

        loop {
            select! {
                _ = cancellation_token.cancelled() => {
                    info!("GPU allocation watcher shutdown requested");
                    break;
                }
                result = self.watch_and_patch_gpu_device_state(&mut previous_device_ids, kubeconfig.clone()) => {
                    match result {
                        Ok(()) => {
                            warn!("GPU allocation watch stream ended unexpectedly, restarting...");
                        }
                        Err(e) => {
                            error!("GPU allocation watch failed: {e:?}");
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
    async fn watch_and_patch_gpu_device_state(
        &self,
        previous_device_ids: &mut HashSet<String>,
        kubeconfig: Option<PathBuf>,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Starting GPU device state watcher");
        let client = kube_client::init_kube_client(kubeconfig).await?;
        let gpu_api: Api<GPU> = Api::all(client);

        let resource_to_system_map = Self::create_resource_system_map();

        info!(
            "Starting GPU device state watcher for path: {:?}",
            self.kubelet_device_state_path
        );

        let grpc_check_result = self.check_grpc_accessibility().await;
        if grpc_check_result.is_err() {
            warn!("gRPC kubelet pod-resources API is not accessible, fallback to checkpoint_file_only_mode for watching other device plugin managed GPUs");
        } else {
            info!("gRPC kubelet pod-resources API is accessible, using gRPC for watching other device plugin managed GPUs");
        }
        let use_grpc = grpc_check_result.is_ok();

        // Set up filesystem watcher
        let (fs_tx, mut fs_rx) = mpsc::channel(10);
        let watcher_result = self.setup_filesystem_watcher(fs_tx).await;

        // Hybrid approach: filesystem events + periodic polling fallback
        let mut poll_interval = interval(Duration::from_secs(30));
        poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let patch_duration_with_jitter = Self::duration_with_jitter(Duration::from_secs(120), 0.15); // ±15% jitter
        let mut patch_all_devices_interval = interval(patch_duration_with_jitter);
        patch_all_devices_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        if let Err(e) = watcher_result {
            warn!("Failed to setup filesystem watcher, falling back to polling only: {e:?}");
            poll_interval = interval(Duration::from_secs(5));
            poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        }

        // Keep watcher alive by holding it in scope
        info!("Filesystem watcher enabled for real-time updates");

        loop {
            select! {
                // Process filesystem events
                Some(_event) = fs_rx.recv() => {
                    debug!("Filesystem event detected, processing device state");
                    if let Err(e) = self.read_and_process_device_state(&gpu_api, previous_device_ids, &resource_to_system_map, false, use_grpc).await {
                        error!("Failed to process device state after filesystem event: {e:?}");
                    }
                }
                // Fallback polling every 30 seconds
                _ = poll_interval.tick() => {
                    debug!("Periodic polling check");
                    if let Err(e) = self.read_and_process_device_state(&gpu_api, previous_device_ids, &resource_to_system_map, false, use_grpc).await {
                        error!("Failed to process device state during periodic check: {e:?}");
                    }
                }

                // Patch all devices periodically with jitter
                _ = patch_all_devices_interval.tick() => {
                    debug!("Checking all devices");
                    if let Err(e) = self.read_and_process_device_state(&gpu_api, previous_device_ids, &resource_to_system_map, true, use_grpc).await {
                        error!("Failed to process device state during patch all devices check: {e:?}");
                    }
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
            message: "Failed to create filesystem watcher".to_string(),
        })?;

        // Spawn task to forward events from sync to async channel
        let fs_tx_clone = fs_tx.clone();
        tokio::task::spawn_blocking(move || {
            while let Ok(event) = rx.recv() {
                let send_result =
                    tokio::runtime::Handle::current().block_on(fs_tx_clone.send(event));
                if send_result.is_err() {
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
        patch_all_devices: bool,
        use_grpc: bool,
    ) -> Result<(), Report<KubernetesError>> {
        // Read and parse the kubelet device state file
        let device_state = self.read_device_state_file().await?;

        // Extract current device IDs from PodDeviceEntries
        let (mut current_allocated_device_ids, current_registered_device_ids) =
            self.extract_device_ids(&device_state, resource_to_system_map)?;

        if use_grpc {
            // grpc result is more accurate and real-time
            // checkpoint file may not updated after Pod deleted
            current_allocated_device_ids = self.get_allocated_devices().await?;
        }

        // Find added and removed devices
        let (added_devices, removed_devices) = if patch_all_devices {
            (
                current_allocated_device_ids.clone(),
                current_registered_device_ids
                    .difference(&current_allocated_device_ids)
                    .cloned()
                    .collect::<HashSet<String>>(),
            )
        } else {
            (
                current_allocated_device_ids
                    .difference(previous_device_ids)
                    .cloned()
                    .collect(),
                previous_device_ids
                    .difference(&current_allocated_device_ids)
                    .cloned()
                    .collect(),
            )
        };

        // Process added devices
        let mut has_error = false;
        for device_id in &added_devices {
            info!("Device added: {}", device_id);
            self.log_device_allocation_details(&device_state, device_id);

            if let Err(e) = self
                .patch_gpu_resource_for_added_device(
                    gpu_api,
                    device_id,
                    &device_state,
                    resource_to_system_map,
                )
                .await
            {
                error!(
                    "Failed to patch GPU resource for added device {}: {e:?}",
                    device_id
                );
                has_error = true;
            }
        }

        // Process removed devices
        for device_id in &removed_devices {
            info!("Device removed: {}", device_id);

            if let Err(e) = self
                .patch_gpu_resource_for_removed_device(gpu_api, device_id)
                .await
            {
                error!(
                    "Failed to patch GPU resource for removed device {}: {e:?}",
                    device_id
                );
                has_error = true;
            }
        }

        // Update previous state when no error, if any error occurred, retry in next loop
        if !has_error {
            *previous_device_ids = current_allocated_device_ids;
        }
        Ok(())
    }

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

    // return (allocated_device_ids, registered_device_ids)
    fn extract_device_ids(
        &self,
        device_state: &KubeletDeviceState,
        resource_to_system_map: &HashMap<String, String>,
    ) -> Result<(HashSet<String>, HashSet<String>), Report<KubernetesError>> {
        let mut allocated_device_ids = HashSet::new();
        let mut registered_device_ids = HashSet::new();

        if let Some(pod_device_entries) = &device_state.data.pod_device_entries {
            for entry in pod_device_entries {
                // just extract GPU devices from resource allocation state
                if resource_to_system_map.contains_key(&entry.resource_name) {
                    for device_list in entry.device_ids.values() {
                        for device_id in device_list {
                            allocated_device_ids.insert(device_id.to_lowercase());
                        }
                    }
                }
            }
        }

        if let Some(device_ids) = device_state.data.registered_devices.get("nvidia.com/gpu") {
            registered_device_ids.extend(device_ids.iter().map(|id| id.to_lowercase()));
        }

        Ok((allocated_device_ids, registered_device_ids))
    }

    fn log_device_allocation_details(&self, device_state: &KubeletDeviceState, device_id: &str) {
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

    async fn patch_gpu_resource_for_added_device(
        &self,
        gpu_api: &Api<GPU>,
        device_id: &str,
        device_state: &KubeletDeviceState,
        resource_to_system_map: &HashMap<String, String>,
    ) -> Result<(), Report<KubernetesError>> {
        // Find the resource name for this device
        let resource_name = self.find_resource_name_for_device(device_state, device_id)?;
        let used_by_system = resource_to_system_map
            .get(&resource_name)
            .unwrap_or(&"nvidia-device-plugin".to_string())
            .clone();

        info!(
            "Patching GPU resource for device {} with usedBySystem: {}",
            device_id, used_by_system
        );

        self.patch_gpu_resource_with_retry(gpu_api, device_id, &used_by_system)
            .await
    }

    async fn patch_gpu_resource_for_removed_device(
        &self,
        gpu_api: &Api<GPU>,
        device_id: &str,
    ) -> Result<(), Report<KubernetesError>> {
        info!(
            "Patching GPU resource for removed device {} with usedBySystem: tensor-fusion",
            device_id
        );

        self.patch_gpu_resource_with_retry(gpu_api, device_id, "tensor-fusion")
            .await
    }

    async fn patch_gpu_resource_with_retry(
        &self,
        gpu_api: &Api<GPU>,
        device_id: &str,
        used_by_system: &str,
    ) -> Result<(), Report<KubernetesError>> {
        const MAX_RETRIES: u32 = 3;
        let mut retry_count = 0;

        while retry_count < MAX_RETRIES {
            match self
                .patch_gpu_resource(gpu_api, device_id, used_by_system)
                .await
            {
                Ok(()) => {
                    info!(
                        "Successfully patched GPU resource for device: {}",
                        device_id
                    );
                    return Ok(());
                }
                Err(e) => {
                    retry_count += 1;
                    warn!(
                        "Failed to patch GPU resource (attempt {}/{}): {e:?}",
                        retry_count, MAX_RETRIES
                    );

                    if retry_count < MAX_RETRIES {
                        let backoff_duration = Duration::from_millis(200 * (1 << retry_count));
                        sleep(backoff_duration).await;
                    }
                }
            }
        }

        Err(Report::new(KubernetesError::WatchFailed {
            message: format!("Failed to patch GPU resource after {MAX_RETRIES} retries"),
        }))
    }

    async fn patch_gpu_resource(
        &self,
        gpu_api: &Api<GPU>,
        device_id: &str,
        used_by_system: &str,
    ) -> Result<(), Report<KubernetesError>> {
        // Get current resource
        let mut current_resource =
            gpu_api
                .get(device_id)
                .await
                .change_context(KubernetesError::WatchFailed {
                    message: format!("Failed to get GPU resource: {device_id}"),
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
        if let Some(status) = &mut current_resource.status {
            status.used_by = Some(used_by_system.to_string());
        }

        // Apply merge patch to status sub-resource
        gpu_api
            .patch_status(
                device_id,
                &kube::api::PatchParams::default(),
                &Patch::Merge(&current_resource),
            )
            .await
            .change_context(KubernetesError::WatchFailed {
                message: format!("Failed to patch GPU resource status: {device_id}"),
            })?;

        Ok(())
    }

    fn find_resource_name_for_device(
        &self,
        device_state: &KubeletDeviceState,
        device_id: &str,
    ) -> Result<String, Report<KubernetesError>> {
        if let Some(pod_device_entries) = &device_state.data.pod_device_entries {
            for entry in pod_device_entries {
                for device_list in entry.device_ids.values() {
                    if device_list.iter().any(|d| d.to_lowercase() == device_id) {
                        return Ok(entry.resource_name.clone());
                    }
                }
            }
        }

        Err(Report::new(KubernetesError::AnnotationParseError {
            message: format!("Could not find resource name for device: {device_id}"),
        }))
    }

    /// Get allocatable resources from kubelet pod-resources API
    #[tracing::instrument(skip(self))]
    pub async fn get_allocated_devices(&self) -> Result<HashSet<String>, Report<KubernetesError>> {
        debug!(
            "Connecting to kubelet pod-resources API at {:?}",
            self.kubelet_socket_path
        );

        // Create a channel that connects to the unix socket
        let channel = self.create_unix_channel().await
            .map_err(|e|KubernetesError::ConnectionFailed {
                message: format!("Failed to connect to kubelet socket, pod-resource kubelet API may not enabled or not accessible: {e}") 
            })?;

        // Create a gRPC client using the generated code
        let mut client =
            pod_resources::pod_resources_lister_client::PodResourcesListerClient::new(channel);

        // Make the List request to get allocated resources
        let request = tonic::Request::new(pod_resources::ListPodResourcesRequest {});

        let response =
            client
                .list(request)
                .await
                .map_err(|e| KubernetesError::ConnectionFailed {
                    message: format!("Failed to list pod resources: {e}"),
                })?;

        let pod_resources_response = response.into_inner();

        // Get the resource names we're interested in from the resource-system map
        let resource_system_map = Self::create_resource_system_map();
        let target_resource_names: HashSet<String> = resource_system_map.keys().cloned().collect();

        // Build allocated device map: resource_name -> list of allocated device_ids
        let mut allocated_devices: HashSet<String> = HashSet::new();

        debug!(
            "Processing {} pods for allocated devices",
            pod_resources_response.pod_resources.len()
        );

        // For each pod and each container, loop over containerDevices
        for pod_resource in &pod_resources_response.pod_resources {
            for container in &pod_resource.containers {
                for device in &container.devices {
                    // Filter by resource_name that exists in create_resource_system_map()
                    if target_resource_names.contains(&device.resource_name) {
                        allocated_devices.extend(device.device_ids.clone());
                        debug!(
                            "Found allocated devices for {}: {} devices in pod {}/{}, container {}",
                            device.resource_name,
                            device.device_ids.len(),
                            pod_resource.namespace,
                            pod_resource.name,
                            container.name
                        );
                    }
                }
            }
        }

        info!(
            "Retrieved allocated devices from device-plugins: {} devices",
            allocated_devices.len()
        );
        Ok(allocated_devices)
    }

    /// Check if the gRPC kubelet pod-resources API is accessible
    /// Returns Ok(()) if accessible, Err if not accessible
    #[tracing::instrument(skip(self))]
    pub async fn check_grpc_accessibility(&self) -> Result<(), Report<KubernetesError>> {
        info!(
            "Checking gRPC accessibility for kubelet pod-resources API at {:?}",
            self.kubelet_socket_path
        );

        // First check if the socket file exists
        if tokio::fs::metadata(&self.kubelet_socket_path)
            .await
            .is_err()
        {
            return Err(KubernetesError::ConnectionFailed {
                message: format!(
                    "Kubelet socket file does not exist: {:?}",
                    self.kubelet_socket_path
                ),
            }
            .into());
        }

        // Try to create a channel connection
        let channel =
            self.create_unix_channel()
                .await
                .map_err(|e| KubernetesError::ConnectionFailed {
                    message: format!("Failed to connect to kubelet socket: {e}"),
                })?;

        // Create a gRPC client and attempt a simple request
        let mut client =
            pod_resources::pod_resources_lister_client::PodResourcesListerClient::new(channel);

        // Make a test request to verify the API is responding
        let request = tonic::Request::new(pod_resources::ListPodResourcesRequest {});

        client
            .list(request)
            .await
            .map_err(|e| KubernetesError::ConnectionFailed {
                message: format!(
                    "gRPC request failed - kubelet pod-resources API not responding: {e}"
                ),
            })?;
        Ok(())
    }

    /// Create a gRPC channel connected to the kubelet unix socket
    async fn create_unix_channel(&self) -> Result<tonic::transport::Channel, io::Error> {
        use hyper_util::rt::TokioIo;
        use tonic::transport::{Endpoint, Uri};
        use tower::service_fn;

        // Convert the unix socket path to a URI that tonic can understand
        let socket_path = self.kubelet_socket_path.clone();

        // Create a channel that connects to the unix socket using TokioIo wrapper
        let channel: tonic::transport::Channel = Endpoint::try_from("http://[::]:50051")
            .map_err(|e| io::Error::other(e.to_string()))?
            .connect_with_connector(service_fn(move |_: Uri| {
                let socket_path = socket_path.clone();
                async move {
                    tokio::net::UnixStream::connect(socket_path)
                        .await
                        .map(TokioIo::new)
                }
            }))
            .await
            .map_err(|e| io::Error::other(e.to_string()))?;
        Ok(channel)
    }

    fn create_resource_system_map() -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "nvidia.com/gpu".to_string(),
            "nvidia-device-plugin".to_string(),
        );
        map.insert("amd.com/gpu".to_string(), "amd-device-plugin".to_string());
        map.insert(
            "intel.com/gpu".to_string(),
            "intel-device-plugin".to_string(),
        );
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
    pod_device_entries: Option<Vec<PodDeviceEntry>>,
    // nvidia.com/gpu -> [GPU-uuid, GPU-uuid-2]
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
#[allow(clippy::upper_case_acronyms)]
struct GPU {
    #[serde(flatten)]
    pub metadata: ObjectMeta,
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

    fn meta(&self) -> &ObjectMeta {
        &self.metadata
    }

    fn meta_mut(&mut self) -> &mut ObjectMeta {
        &mut self.metadata
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use similar_asserts::assert_eq;
    use tempfile::NamedTempFile;

    use super::*;

    const KUBELET_DEVICE_STATE_PATH: &str = "/var/lib/kubelet/pod-resources/kubelet.sock";
    fn create_test_device_state() -> KubeletDeviceState {
        let mut device_ids = HashMap::new();
        device_ids.insert(
            "-1".to_string(),
            vec!["GPU-7d8429d5-531d-d6a6-6510-3b662081a75a".to_string()],
        );

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
            vec!["GPU-7d8429d5-531d-d6a6-6510-3b662081a75a".to_string()],
        );

        KubeletDeviceState {
            data: DeviceStateData {
                pod_device_entries: Some(vec![pod_entry]),
                registered_devices,
            },
            checksum: 2262205670,
        }
    }

    fn create_empty_device_state() -> KubeletDeviceState {
        KubeletDeviceState {
            data: DeviceStateData {
                pod_device_entries: None,
                registered_devices: HashMap::new(),
            },
            checksum: 0,
        }
    }

    async fn create_temp_device_state_file(device_state: &KubeletDeviceState) -> NamedTempFile {
        let mut temp_file = NamedTempFile::new().expect("should create temp file");
        let json_content =
            serde_json::to_string(device_state).expect("should serialize device state");
        use std::io::Write;
        temp_file
            .write_all(json_content.as_bytes())
            .expect("should write to temp file");
        temp_file.flush().expect("should flush temp file");
        temp_file
    }

    #[tokio::test]
    async fn test_read_device_state_file_success() {
        let device_state = create_test_device_state();
        let temp_file = create_temp_device_state_file(&device_state).await;

        let watcher =
            GpuDeviceStateWatcher::new(temp_file.path().to_path_buf(), KUBELET_DEVICE_STATE_PATH);
        let result = watcher.read_device_state_file().await;

        assert!(result.is_ok(), "should successfully read device state file");
        let parsed_state = result.unwrap();
        assert_eq!(
            parsed_state.checksum, device_state.checksum,
            "checksum should match"
        );
        assert_eq!(
            parsed_state.data.pod_device_entries.unwrap().len(),
            1,
            "should have one pod device entry"
        );
    }

    #[tokio::test]
    async fn test_read_device_state_file_not_found() {
        let watcher = GpuDeviceStateWatcher::new(
            PathBuf::from("/nonexistent/path"),
            KUBELET_DEVICE_STATE_PATH,
        );
        let result = watcher.read_device_state_file().await;

        assert!(result.is_err(), "should fail when file does not exist");
        let error = result.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Failed to read device state file"),
            "error should mention file read failure"
        );
    }

    #[tokio::test]
    async fn test_read_device_state_file_invalid_json() {
        let mut temp_file = NamedTempFile::new().expect("should create temp file");
        use std::io::Write;
        temp_file
            .write_all(b"invalid json content")
            .expect("should write to temp file");
        temp_file.flush().expect("should flush temp file");

        let watcher =
            GpuDeviceStateWatcher::new(temp_file.path().to_path_buf(), KUBELET_DEVICE_STATE_PATH);
        let result = watcher.read_device_state_file().await;

        assert!(result.is_err(), "should fail when JSON is invalid");
        let error = result.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Failed to parse device state JSON"),
            "error should mention JSON parse failure"
        );
    }

    #[test]
    fn test_extract_device_ids_with_devices() {
        let watcher = GpuDeviceStateWatcher::new("/test", KUBELET_DEVICE_STATE_PATH);
        let device_state = create_test_device_state();

        let resource_system_map = GpuDeviceStateWatcher::create_resource_system_map();
        let result = watcher.extract_device_ids(&device_state, &resource_system_map);

        assert!(result.is_ok(), "should successfully extract device IDs");
        let device_ids = result.unwrap();
        assert_eq!(device_ids.0.len(), 1, "should extract one device ID");
        assert!(
            device_ids
                .0
                .contains("gpu-7d8429d5-531d-d6a6-6510-3b662081a75a"),
            "should contain expected device ID"
        );
    }

    #[test]
    fn test_extract_device_ids_empty() {
        let watcher = GpuDeviceStateWatcher::new("/test", KUBELET_DEVICE_STATE_PATH);
        let device_state = create_empty_device_state();

        let resource_system_map = GpuDeviceStateWatcher::create_resource_system_map();
        let result = watcher.extract_device_ids(&device_state, &resource_system_map);

        assert!(
            result.is_ok(),
            "should successfully extract empty device IDs"
        );
        let device_ids = result.unwrap();
        assert_eq!(device_ids.0.len(), 0, "should extract no device IDs");
    }

    #[test]
    fn test_extract_device_ids_multiple_devices() {
        let watcher = GpuDeviceStateWatcher::new("/test", KUBELET_DEVICE_STATE_PATH);

        let mut device_ids_1 = HashMap::new();
        device_ids_1.insert(
            "-1".to_string(),
            vec!["GPU-1".to_string(), "GPU-2".to_string()],
        );

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
                pod_device_entries: Some(vec![pod_entry_1, pod_entry_2]),
                registered_devices: HashMap::new(),
            },
            checksum: 12345,
        };

        let resource_system_map = GpuDeviceStateWatcher::create_resource_system_map();
        let result = watcher.extract_device_ids(&device_state, &resource_system_map);

        assert!(
            result.is_ok(),
            "should successfully extract multiple device IDs"
        );
        let device_ids = result.unwrap();
        assert_eq!(
            device_ids.0.len(),
            3,
            "should extract three unique device IDs"
        );
        assert!(device_ids.0.contains("gpu-1"), "should contain GPU-1");
        assert!(device_ids.0.contains("gpu-2"), "should contain GPU-2");
        assert!(device_ids.0.contains("gpu-3"), "should contain GPU-3");
    }

    #[test]
    fn test_find_resource_name_for_device_success() {
        let watcher = GpuDeviceStateWatcher::new("/test", KUBELET_DEVICE_STATE_PATH);
        let device_state = create_test_device_state();

        let result = watcher.find_resource_name_for_device(
            &device_state,
            "gpu-7d8429d5-531d-d6a6-6510-3b662081a75a",
        );

        assert!(result.is_ok(), "should successfully find resource name");
        let resource_name = result.unwrap();
        assert_eq!(
            resource_name, "nvidia.com/gpu",
            "should return correct resource name"
        );
    }

    #[test]
    fn test_find_resource_name_for_device_not_found() {
        let watcher = GpuDeviceStateWatcher::new("/test", KUBELET_DEVICE_STATE_PATH);
        let device_state = create_test_device_state();

        let result = watcher.find_resource_name_for_device(&device_state, "nonexistent-device");

        assert!(result.is_err(), "should fail when device not found");
        let error = result.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Could not find resource name for device"),
            "error should mention device not found"
        );
    }

    #[test]
    fn test_create_resource_system_map() {
        let map = GpuDeviceStateWatcher::create_resource_system_map();

        assert_eq!(map.len(), 3, "should contain three resource mappings");
        assert_eq!(
            map.get("nvidia.com/gpu"),
            Some(&"nvidia-device-plugin".to_string()),
            "should map nvidia.com/gpu correctly"
        );
        assert_eq!(
            map.get("amd.com/gpu"),
            Some(&"amd-device-plugin".to_string()),
            "should map amd.com/gpu correctly"
        );
        assert_eq!(
            map.get("intel.com/gpu"),
            Some(&"intel-device-plugin".to_string()),
            "should map intel.com/gpu correctly"
        );
    }

    #[test]
    fn test_log_device_allocation_details() {
        let watcher = GpuDeviceStateWatcher::new("/test", KUBELET_DEVICE_STATE_PATH);
        let device_state = create_test_device_state();

        // This test mainly ensures the function doesn't panic
        // In a real scenario, we would capture log output to verify the content
        watcher.log_device_allocation_details(
            &device_state,
            "GPU-7d8429d5-531d-d6a6-6510-3b662081a75a",
        );
    }

    #[test]
    fn test_kubelet_device_state_serialization() {
        let device_state = create_test_device_state();

        // Test serialization
        let json_str = serde_json::to_string(&device_state).expect("should serialize to JSON");
        assert!(
            json_str.contains("PodDeviceEntries"),
            "JSON should contain PodDeviceEntries"
        );
        assert!(
            json_str.contains("RegisteredDevices"),
            "JSON should contain RegisteredDevices"
        );

        // Test deserialization
        let deserialized: KubeletDeviceState =
            serde_json::from_str(&json_str).expect("should deserialize from JSON");
        assert_eq!(
            deserialized.checksum, device_state.checksum,
            "checksum should match after round-trip"
        );
        assert_eq!(
            deserialized.data.pod_device_entries.unwrap().len(),
            device_state.data.pod_device_entries.unwrap().len(),
            "pod entries count should match"
        );
    }

    #[test]
    fn test_gpu_resource_status_default() {
        let status = GpuResourceStatus::default();
        assert_eq!(
            status.used_by, None,
            "default status should have None for used_by"
        );
    }

    #[test]
    fn test_gpu_resource_serialization() {
        let status = GpuResourceStatus {
            used_by: Some("nvidia-device-plugin".to_string()),
        };

        let json_str =
            serde_json::to_string(&status).expect("should serialize GPU resource status");
        assert!(
            json_str.contains("usedBy"),
            "JSON should use camelCase for used_by field"
        );

        let deserialized: GpuResourceStatus =
            serde_json::from_str(&json_str).expect("should deserialize GPU resource status");
        assert_eq!(
            deserialized.used_by,
            Some("nvidia-device-plugin".to_string()),
            "used_by should match after round-trip"
        );
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
        assert!(
            result.is_ok(),
            "should successfully deserialize GPU resource without spec field"
        );

        let gpu = result.unwrap();
        assert!(
            gpu.spec.is_none(),
            "spec should be None when not present in JSON"
        );
        assert!(gpu.status.is_some(), "status should be present");
        assert_eq!(
            gpu.status.unwrap().used_by,
            Some("nvidia-device-plugin".to_string()),
            "used_by should match"
        );
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
        assert!(
            result.is_ok(),
            "should successfully deserialize GPU resource with spec field"
        );

        let gpu = result.unwrap();
        assert!(
            gpu.spec.is_some(),
            "spec should be present when included in JSON"
        );
        assert_eq!(
            gpu.spec.unwrap().dummy,
            Some("test-value".to_string()),
            "dummy field should match"
        );
        assert!(gpu.status.is_some(), "status should be present");
        assert_eq!(
            gpu.status.unwrap().used_by,
            Some("nvidia-device-plugin".to_string()),
            "used_by should match"
        );
    }

    #[test]
    fn test_device_state_with_complex_device_ids() {
        let watcher = GpuDeviceStateWatcher::new("/test", KUBELET_DEVICE_STATE_PATH);

        // Create a more complex device ID structure
        let mut device_ids = HashMap::new();
        device_ids.insert("-1".to_string(), vec!["GPU-1".to_string()]);
        device_ids.insert(
            "0".to_string(),
            vec!["GPU-2".to_string(), "GPU-3".to_string()],
        );

        let pod_entry = PodDeviceEntry {
            pod_uid: "complex-pod".to_string(),
            container_name: "complex-container".to_string(),
            resource_name: "nvidia.com/gpu".to_string(),
            device_ids,
            alloc_resp: "complex-alloc".to_string(),
        };

        let device_state = KubeletDeviceState {
            data: DeviceStateData {
                pod_device_entries: Some(vec![pod_entry]),
                registered_devices: HashMap::new(),
            },
            checksum: 54321,
        };

        let resource_system_map = GpuDeviceStateWatcher::create_resource_system_map();
        let result = watcher.extract_device_ids(&device_state, &resource_system_map);
        assert!(result.is_ok(), "should handle complex device ID structure");

        let device_ids = result.unwrap();
        assert_eq!(device_ids.0.len(), 3, "should extract all three device IDs");
        assert!(device_ids.0.contains("gpu-1"), "should contain GPU-1");
        assert!(device_ids.0.contains("gpu-2"), "should contain GPU-2");
        assert!(device_ids.0.contains("gpu-3"), "should contain GPU-3");
    }
}
