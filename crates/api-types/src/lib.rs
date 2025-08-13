//! Shared API type definitions
//!
//! This crate contains shared API type definitions used across different components
//! in the vgpu project, including Pod resource information, query response formats,
//! and related Kubernetes authentication types.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Copy, Serialize, Deserialize)]
pub enum QosLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for QosLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Worker resource information including requests and limits
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkerInfo {
    /// Pod name
    pub pod_name: String,
    /// Pod namespace
    pub namespace: String,
    /// Node name where the pod is scheduled
    pub node_name: Option<String>,
    /// TFLOPS request value
    pub tflops_request: Option<f64>,
    /// VRAM request value in bytes
    pub vram_request: Option<u64>,
    /// TFLOPS limit value
    pub tflops_limit: Option<f64>,
    /// VRAM limit value in bytes
    pub vram_limit: Option<u64>,
    /// List of GPU UUIDs
    pub gpu_uuids: Option<Vec<String>>,
    /// QoS level for the workload
    pub qos_level: Option<QosLevel>,
    /// Container names
    pub containers: Option<Vec<String>>,
    /// Host pid
    pub host_pid: u32,
    /// Workload name
    pub workload_name: Option<String>,
    /// Pod labels
    pub labels: BTreeMap<String, String>,
}

/// Response format for Worker query API (legacy, use PodInfoResponse for new code)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResponse {
    /// Whether the request was successful
    pub success: bool,
    /// Pod resource information data (present when successful)
    pub data: Option<WorkerInfo>,
    /// Response message
    pub message: String,
}

/// Response for pod information (GPU and limiter configuration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodInfoResponse {
    /// Whether the request was successful
    pub success: bool,
    /// Pod information data (present when successful)
    pub data: Option<PodInfo>,
    /// Response message
    pub message: String,
}

/// Pod-level information for GPU and limiter setup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodInfo {
    /// Pod name
    pub pod_name: String,
    /// Pod namespace
    pub namespace: String,
    /// List of GPU UUIDs assigned to this pod
    pub gpu_uuids: Vec<String>,
    /// TFLOPS limit for the pod
    pub tflops_limit: Option<f64>,
    /// VRAM limit for the pod in bytes
    pub vram_limit: Option<u64>,
    /// QoS level for the workload
    pub qos_level: Option<QosLevel>,
}

/// Response for process initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInitResponse {
    /// Whether the request was successful
    pub success: bool,
    /// Process information data (present when successful)
    pub data: Option<ProcessInfo>,
    /// Response message
    pub message: String,
}

/// Process-level information after successful initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    /// Host PID of the process
    pub host_pid: u32,
    /// Container PID of the process
    pub container_pid: u32,
    /// Container name
    pub container_name: String,
    /// Pod name
    pub pod_name: String,
    /// Namespace
    pub namespace: String,
}

/// Kubernetes information from JWT token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesInfo {
    pub namespace: String,

    pub node: Option<KubernetesNode>,

    pub pod: KubernetesPod,

    pub serviceaccount: KubernetesServiceAccount,
}

/// Kubernetes node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesNode {
    pub name: String,
    pub uid: String,
}

/// Kubernetes Pod information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesPod {
    pub name: String,
    pub uid: String,
}

/// Kubernetes service account information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesServiceAccount {
    pub name: String,
    pub uid: String,
}

/// JWT payload structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtPayload {
    #[serde(rename = "kubernetes.io")]
    pub kubernetes: KubernetesInfo,
    pub nbf: i64,
    pub sub: String,
}

/// JWT authentication configuration
#[derive(Debug, Clone)]
pub struct JwtAuthConfig {
    pub public_key: String,
}

/// command type, sent by Hypervisor to Limiter
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LimiterCommandType {
    /// TensorFusion health check
    TfHealthCheck,
    /// suspend TensorFusion
    TfSuspend,
    /// resume TensorFusion
    TfResume,
    /// reclaim VRAM
    TfVramReclaim,
}

/// command sent by Hypervisor to Limiter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimiterCommand {
    /// command unique ID
    pub id: u64,
    /// command type
    pub kind: LimiterCommandType,
}

/// result of command execution by Limiter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimiterCommandResponse {
    /// command ID
    pub id: u64,
    /// whether the command was successful
    pub success: bool,
    /// optional description
    pub message: Option<String>,
}
