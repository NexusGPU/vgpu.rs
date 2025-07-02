//! Shared API type definitions
//!
//! This crate contains shared API type definitions used across different components
//! in the vgpu project, including Pod resource information, query response formats,
//! and related Kubernetes authentication types.

use serde::Deserialize;
use serde::Serialize;

/// Pod resource information including requests and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodResourceInfo {
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
}

/// Response format for Pod query API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodQueryResponse {
    /// Whether the request was successful
    pub success: bool,
    /// Pod resource information data (present when successful)
    pub data: Option<PodResourceInfo>,
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
