use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use error_stack::Report;
use poem::get;
use poem::handler;
use poem::listener::TcpListener;
use poem::middleware::Tracing;
use poem::web::Path;
use poem::EndpointExt;
use poem::Route;
use poem::Server;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::oneshot;
use tracing::error;
use tracing::info;

use crate::k8s::annotations::TensorFusionAnnotations;

/// Pod resource information including requests and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodResourceInfo {
    /// Pod name
    pub pod_name: String,
    /// Pod namespace
    pub namespace: String,
    /// Node name where the pod is scheduled
    pub node_name: Option<String>,
    /// TFLOPS request
    pub tflops_request: Option<f64>,
    /// VRAM request in bytes
    pub vram_request: Option<u64>,
    /// TFLOPS limit
    pub tflops_limit: Option<f64>,
    /// VRAM limit in bytes
    pub vram_limit: Option<u64>,
}

impl From<(String, String, Option<String>, TensorFusionAnnotations)> for PodResourceInfo {
    fn from(
        (pod_name, namespace, node_name, annotations): (
            String,
            String,
            Option<String>,
            TensorFusionAnnotations,
        ),
    ) -> Self {
        Self {
            pod_name,
            namespace,
            node_name,
            tflops_request: annotations.tflops_request,
            vram_request: annotations.vram_request,
            tflops_limit: annotations.tflops_limit,
            vram_limit: annotations.vram_limit,
        }
    }
}

/// API response for pod query
#[derive(Debug, Serialize)]
pub struct PodQueryResponse {
    pub success: bool,
    pub data: Option<PodResourceInfo>,
    pub message: String,
}

/// API errors
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Pod not found: {pod_name} in namespace {namespace}")]
    #[allow(dead_code)]
    PodNotFound { pod_name: String, namespace: String },
    #[error("Server error: {message}")]
    ServerError { message: String },
}

/// Pod storage for API queries
pub type PodStorage = Arc<RwLock<HashMap<String, PodResourceInfo>>>;

/// HTTP API server for querying pod resource information
pub struct ApiServer {
    pod_storage: PodStorage,
    listen_addr: String,
}

impl ApiServer {
    /// Create a new API server
    pub fn new(pod_storage: PodStorage, listen_addr: String) -> Self {
        Self {
            pod_storage,
            listen_addr,
        }
    }

    /// Start the API server
    ///
    /// # Errors
    ///
    /// - [`ApiError::ServerError`] if the server fails to start or bind to the address
    pub async fn run(self, mut shutdown_rx: oneshot::Receiver<()>) -> Result<(), Report<ApiError>> {
        info!("Starting HTTP API server on {}", self.listen_addr);

        let app = Route::new()
            .at("/api/v1/pods/:namespace/:pod_name", get(get_pod_info))
            .data(self.pod_storage)
            .with(Tracing);

        let listener = TcpListener::bind(&self.listen_addr);
        let server = Server::new(listener);

        tokio::select! {
            result = server.run(app) => {
                match result {
                    Ok(()) => {
                        info!("API server stopped normally");
                        Ok(())
                    }
                    Err(e) => {
                        error!("API server failed: {e}");
                        Err(Report::new(ApiError::ServerError {
                            message: format!("Server failed: {e}"),
                        }))
                    }
                }
            }
            _ = &mut shutdown_rx => {
                info!("API server shutdown requested");
                Ok(())
            }
        }
    }
}

/// Get pod resource information
#[handler]
async fn get_pod_info(
    Path((namespace, pod_name)): Path<(String, String)>,
    pod_storage: poem::web::Data<&PodStorage>,
) -> poem::Result<poem::web::Json<PodQueryResponse>> {
    let pod_key = format!("{namespace}/{pod_name}");

    let storage = pod_storage.read().map_err(|e| {
        error!("Failed to acquire read lock: {e}");
        poem::Error::from_string(
            "Internal server error",
            poem::http::StatusCode::INTERNAL_SERVER_ERROR,
        )
    })?;

    match storage.get(&pod_key) {
        Some(pod_info) => {
            info!("Found pod info for {pod_key}");
            Ok(poem::web::Json(PodQueryResponse {
                success: true,
                data: Some(pod_info.clone()),
                message: "Pod found".to_string(),
            }))
        }
        None => {
            info!("Pod not found: {pod_key}");
            Ok(poem::web::Json(PodQueryResponse {
                success: false,
                data: None,
                message: format!("Pod not found: {pod_name} in namespace {namespace}"),
            }))
        }
    }
}

/// Update pod information in storage
pub fn update_pod_storage(
    storage: &PodStorage,
    pod_name: String,
    namespace: String,
    node_name: Option<String>,
    annotations: TensorFusionAnnotations,
) {
    let pod_key = format!("{namespace}/{pod_name}");
    let pod_info = PodResourceInfo::from((pod_name, namespace, node_name, annotations));

    match storage.write() {
        Ok(mut storage_guard) => {
            storage_guard.insert(pod_key.clone(), pod_info);
            info!("Updated pod storage for {pod_key}");
        }
        Err(e) => {
            error!("Failed to acquire write lock for pod storage: {e}");
        }
    }
}

/// Remove pod information from storage
pub fn remove_pod_from_storage(storage: &PodStorage, pod_name: &str, namespace: &str) {
    let pod_key = format!("{namespace}/{pod_name}");

    match storage.write() {
        Ok(mut storage_guard) => {
            if storage_guard.remove(&pod_key).is_some() {
                info!("Removed pod from storage: {pod_key}");
            }
        }
        Err(e) => {
            error!("Failed to acquire write lock for pod storage removal: {e}");
        }
    }
}
