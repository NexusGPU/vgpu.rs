//! Types and data structures for GPU device state watching

use std::collections::HashMap;
use k8s_openapi::ClusterResourceScope;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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