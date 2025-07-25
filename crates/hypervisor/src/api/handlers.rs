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

use super::types::JwtPayload;
use super::types::PodInfo;
use super::types::PodInfoResponse;
use super::types::ProcessInfo;
use super::types::ProcessInitResponse;
use crate::gpu_observer::GpuObserver;

use crate::pod_management::PodManager;

/// Query parameters for process initialization
#[derive(Debug, Deserialize)]
pub struct ProcessInitQuery {
    pub container_name: String,
    pub container_pid: u32,
}

/// Get pod-level GPU and limiter information using JWT token
#[handler]
pub async fn get_pod_info(
    req: &Request,
    pod_manager: Data<&Arc<PodManager>>,
) -> poem::Result<poem::web::Json<PodInfoResponse>> {
    // Extract JWT payload from request extensions
    let jwt_payload = req.extensions().get::<JwtPayload>().ok_or_else(|| {
        poem::Error::from_string(
            "JWT payload not found in request",
            poem::http::StatusCode::UNAUTHORIZED,
        )
    })?;
    let pod_name = &jwt_payload.kubernetes.pod.name;
    let namespace = &jwt_payload.kubernetes.namespace;

    let Some(pod_entry) = pod_manager.find_pod_by_name(namespace, pod_name).await else {
        warn!(
            pod_name = pod_name,
            namespace = namespace,
            "Pod not found in registry"
        );
        return Ok(poem::web::Json(PodInfoResponse {
            success: false,
            data: None,
            message: format!("Pod {pod_name} not found in namespace {namespace}"),
        }));
    };

    info!(pod_name = pod_name, "Pod found in registry");

    let pod_info = PodInfo {
        pod_name: pod_entry.info.pod_name.clone(),
        namespace: pod_entry.info.namespace.clone(),
        gpu_uuids: pod_entry.info.gpu_uuids.clone().unwrap_or_default(),
        tflops_limit: pod_entry.info.tflops_limit,
        vram_limit: pod_entry.info.vram_limit,
        qos_level: pod_entry.info.qos_level,
    };

    Ok(poem::web::Json(PodInfoResponse {
        success: true,
        data: Some(pod_info),
        message: format!("Pod {pod_name} information retrieved successfully"),
    }))
}

/// Initialize a CUDA process in a container
#[handler]
pub async fn process_init(
    req: &Request,
    query: Query<ProcessInitQuery>,
    pod_manager: Data<&Arc<PodManager>>,
    gpu_observer: Data<&Arc<GpuObserver>>,
) -> poem::Result<poem::web::Json<ProcessInitResponse>> {
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

    // Validate pod and container exist
    let gpu_uuids = {
        let Some(pod_entry) = pod_manager.find_pod_by_name(namespace, pod_name).await else {
            warn!(pod_name = pod_name, namespace = namespace, "Pod not found");
            return Ok(poem::web::Json(ProcessInitResponse {
                success: false,
                data: None,
                message: format!("Pod {pod_name} not found in namespace {namespace}"),
            }));
        };

        if pod_entry.get_container(container_name).is_none() {
            warn!(
                pod_name = pod_name,
                container_name = container_name,
                "Container not found"
            );
            return Ok(poem::web::Json(ProcessInitResponse {
                success: false,
                data: None,
                message: format!("Container {container_name} not found in pod {pod_name}"),
            }));
        }

        pod_entry.info.gpu_uuids.clone().unwrap_or_default()
    };

    // Initialize the process (discover PID and register to all components)
    let discovery_timeout = Duration::from_secs(5);
    let process_result = match timeout(
        discovery_timeout,
        pod_manager.initialize_process(
            pod_name,
            namespace,
            container_name,
            container_pid,
            gpu_observer.clone(),
        ),
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
                gpu_uuids,
            }
        }
        Ok(Err(e)) => {
            warn!(error = %e, "Failed to initialize process");
            return Ok(poem::web::Json(ProcessInitResponse {
                success: false,
                data: None,
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
                data: None,
                message: format!(
                    "Process initialization timed out after {} seconds",
                    discovery_timeout.as_secs()
                ),
            }));
        }
    };

    Ok(poem::web::Json(ProcessInitResponse {
        success: true,
        data: Some(process_result),
        message: "Process initialized successfully".to_string(),
    }))
}
