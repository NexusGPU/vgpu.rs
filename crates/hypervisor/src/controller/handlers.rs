use std::sync::Arc;
use std::time::Duration;

use poem::handler;
use poem::web::Data;
use poem::web::Query;
use poem::Request;
use serde::Deserialize;
use tokio::time::timeout;
use tracing::info;
use tracing::warn;

use super::JwtPayload;
use super::PodInfo;
use super::PodInfoResponse;
use super::ProcessInfo;
use super::ProcessInitResponse;

use crate::config::AutoFreezeAndResume;
use crate::core::pod::traits::{DeviceSnapshotProvider, PodStateRepository, TimeSource};
use crate::core::pod::PodManager;
use api_types::AutoFreezeConfig;
use utils::shared_memory::traits::SharedMemoryAccess;

/// Query parameters for pod info request
#[derive(Debug, Deserialize)]
pub struct PodInfoQuery {
    pub container_name: Option<String>,
}

/// Query parameters for process initialization
#[derive(Debug, Deserialize)]
pub struct ProcessInitQuery {
    pub container_name: String,
    pub container_pid: u32,
}

/// Get pod-level GPU and limiter information using JWT token
#[handler]
pub async fn get_pod_info<M, P, D, T>(
    req: &Request,
    query: Query<PodInfoQuery>,
    pod_manager: Data<&Arc<PodManager<M, P, D, T>>>,
    auto_freeze_config: Data<&Arc<AutoFreezeAndResume>>,
) -> poem::Result<poem::web::Json<PodInfoResponse>>
where
    M: SharedMemoryAccess + 'static,
    P: PodStateRepository + 'static,
    D: DeviceSnapshotProvider + 'static,
    T: TimeSource + 'static,
{
    // Extract JWT payload from request extensions
    let jwt_payload = req.extensions().get::<JwtPayload>().ok_or_else(|| {
        poem::Error::from_string(
            "JWT payload not found in request",
            poem::http::StatusCode::UNAUTHORIZED,
        )
    })?;
    let pod_name = &jwt_payload.kubernetes.pod.name;
    let namespace = &jwt_payload.kubernetes.namespace;

    let pod_entry = match pod_manager.find_pod_by_name(namespace, pod_name).await {
        Ok(Some(pod_entry)) => pod_entry,
        Ok(None) => {
            warn!(
                pod_name = pod_name,
                namespace = namespace,
                "Pod not found in registry"
            );
            return Ok(poem::web::Json(PodInfoResponse {
                success: false,
                pod_info: None,
                message: format!("Pod {pod_name} not found in namespace {namespace}"),
            }));
        }
        Err(e) => {
            warn!(error = %e, "Failed to find pod in registry");
            return Ok(poem::web::Json(PodInfoResponse {
                success: false,
                pod_info: None,
                message: format!("Failed to find pod in registry: {e}"),
            }));
        }
    };

    let auto_freeze = pod_entry.qos_level.and_then(|qos| {
        auto_freeze_config
            .auto_freeze
            .iter()
            .find(|config| config.qos == qos)
            .map(|config| AutoFreezeConfig {
                freeze_to_mem_ttl: config.freeze_to_mem_ttl.clone(),
                freeze_to_disk_ttl: config.freeze_to_disk_ttl.clone(),
                enable: config.enable.unwrap_or(false),
            })
    });

    // Select GPU UUIDs based on container name if provided
    let gpu_uuids = if let Some(container_name) = &query.container_name {
        // Try to get container-specific GPU list
        if let Some(container_gpu_map) = &pod_entry.container_gpu_uuids {
            if let Some(container_gpus) = container_gpu_map.get(container_name) {
                container_gpus
                    .iter()
                    .map(|uuid| uuid.replace("gpu-", "GPU-"))
                    .collect()
            } else {
                // Container name not found in mapping, fallback to pod-level gpu_uuids
                pod_entry
                    .gpu_uuids
                    .map(|uuids| {
                        uuids
                            .into_iter()
                            .map(|uuid| uuid.replace("gpu-", "GPU-"))
                            .collect()
                    })
                    .unwrap_or_default()
            }
        } else {
            // No container-level mapping, use pod-level gpu_uuids
            pod_entry
                .gpu_uuids
                .map(|uuids| {
                    uuids
                        .into_iter()
                        .map(|uuid| uuid.replace("gpu-", "GPU-"))
                        .collect()
                })
                .unwrap_or_default()
        }
    } else {
        // No container_name provided, use pod-level gpu_uuids
        pod_entry
            .gpu_uuids
            .map(|uuids| {
                uuids
                    .into_iter()
                    .map(|uuid| uuid.replace("gpu-", "GPU-"))
                    .collect()
            })
            .unwrap_or_default()
    };

    let pod_info = PodInfo {
        pod_name: pod_entry.pod_name.clone(),
        namespace: pod_entry.namespace.clone(),
        gpu_uuids,
        tflops_limit: pod_entry.tflops_limit,
        vram_limit: pod_entry.vram_limit,
        qos_level: pod_entry.qos_level.map(|qos| qos.into()),
        compute_shard: pod_entry.compute_shard,
        isolation: pod_entry.isolation.clone(),
        auto_freeze,
    };

    pod_manager
        .ensure_pod_registered(namespace, pod_name)
        .await
        .map_err(|e| {
            poem::Error::from_string(e.to_string(), poem::http::StatusCode::INTERNAL_SERVER_ERROR)
        })?;

    Ok(poem::web::Json(PodInfoResponse {
        success: true,
        pod_info: Some(pod_info),
        message: format!("Pod {pod_name} information retrieved successfully"),
    }))
}

/// Initialize a CUDA process in a container
#[handler]
pub async fn process_init<M, P, D, T>(
    req: &Request,
    query: Query<ProcessInitQuery>,
    pod_manager: Data<&Arc<PodManager<M, P, D, T>>>,
) -> poem::Result<poem::web::Json<ProcessInitResponse>>
where
    M: SharedMemoryAccess + 'static,
    P: PodStateRepository + 'static,
    D: DeviceSnapshotProvider + 'static,
    T: TimeSource + 'static,
{
    // Extract JWT payload from request extensions
    let jwt_payload = req.extensions().get::<JwtPayload>().ok_or_else(|| {
        poem::Error::from_string(
            "JWT payload not found in request",
            poem::http::StatusCode::UNAUTHORIZED,
        )
    })?;

    let pod_name = &jwt_payload.kubernetes.pod.name;
    let namespace = &jwt_payload.kubernetes.namespace;
    let container_name = &query.container_name;
    let container_pid = query.container_pid;

    info!(
        pod_name = pod_name,
        namespace = namespace,
        container_name = container_name,
        container_pid = container_pid,
        "Initializing worker"
    );

    // Initialize the process (discover PID and register to all components)
    let discovery_timeout = Duration::from_secs(5);
    let process_result = match timeout(
        discovery_timeout,
        pod_manager.initialize_process(pod_name, namespace, container_name, container_pid),
    )
    .await
    {
        Ok(Ok(host_pid)) => {
            info!(
                pod_name = pod_name,
                container_name = container_name,
                container_pid = container_pid,
                host_pid = host_pid,
                "CUDA process initialized successfully"
            );

            ProcessInfo {
                host_pid,
                container_pid,
                container_name: container_name.to_string(),
                pod_name: pod_name.to_string(),
                namespace: namespace.to_string(),
            }
        }
        Ok(Err(e)) => {
            warn!(error = %e, "Failed to initialize process");
            return Ok(poem::web::Json(ProcessInitResponse {
                success: false,
                process_info: None,
                message: format!("Failed to initialize process: {e}"),
            }));
        }
        Err(_) => {
            warn!(
                timeout_secs = discovery_timeout.as_secs(),
                "Process initialization timed out"
            );
            return Ok(poem::web::Json(ProcessInitResponse {
                success: false,
                process_info: None,
                message: format!(
                    "Process initialization timed out after {} seconds",
                    discovery_timeout.as_secs()
                ),
            }));
        }
    };

    Ok(poem::web::Json(ProcessInitResponse {
        success: true,
        process_info: Some(process_result),
        message: "Process initialized successfully".to_string(),
    }))
}

#[handler]
pub async fn ping() -> &'static str {
    "ok"
}
