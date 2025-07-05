use poem::handler;
use poem::web::Data;
use poem::Request;
use tracing::error;
use tracing::info;

use super::types::JwtPayload;
use crate::api::types::WorkerQueryResponse;
use crate::worker_manager::WorkerRegistry;

/// Core logic for getting pod resource information from WorkerRegistry
async fn get_worker_info_from_registry_impl(
    jwt_payload: &JwtPayload,
    worker_registry: &WorkerRegistry,
) -> poem::Result<WorkerQueryResponse> {
    let pod_name = &jwt_payload.kubernetes.pod.name;
    let namespace = &jwt_payload.kubernetes.namespace;

    info!(
        pod_name = pod_name,
        namespace = namespace,
        "Querying worker info from registry"
    );

    if let Some(worker_entry) = worker_registry.read().await.get(pod_name) {
        info!(pod_name = pod_name, "Worker found in registry");
        Ok(WorkerQueryResponse {
            success: true,
            data: Some(worker_entry.info.clone()),
            message: "Worker information retrieved successfully".to_string(),
        })
    } else {
        info!(pod_name = pod_name, "Worker not found in registry");
        Ok(WorkerQueryResponse {
            success: false,
            data: None,
            message: format!("Worker {pod_name} not found in namespace {namespace}"),
        })
    }
}

/// Get pod resource information
#[handler]
pub async fn get_worker_info(
    req: &Request,
    worker_registry: Data<&WorkerRegistry>,
) -> poem::Result<poem::web::Json<WorkerQueryResponse>> {
    // Extract JWT payload from request extensions
    let jwt_payload = req.extensions().get::<JwtPayload>().ok_or_else(|| {
        error!("JWT payload not found in request extensions");
        poem::Error::from_string(
            "Authentication information missing",
            poem::http::StatusCode::INTERNAL_SERVER_ERROR,
        )
    })?;

    let response = get_worker_info_from_registry_impl(jwt_payload, &worker_registry).await?;
    Ok(poem::web::Json(response))
}
