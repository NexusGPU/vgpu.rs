//! Shared API type definitions
//!
//! This crate contains shared API type definitions used across different components
//! in the vgpu project, including Pod resource information, query response formats,
//! and related Kubernetes authentication types.

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
}

/// Response format for Worker query API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerQueryResponse {
    /// Whether the request was successful
    pub success: bool,
    /// Pod resource information data (present when successful)
    pub data: Option<WorkerInfo>,
    /// Response message
    pub message: String,
}

/// Kubernetes information from JWT token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesInfo {
    pub namespace: String,
    pub node: KubernetesNode,
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
