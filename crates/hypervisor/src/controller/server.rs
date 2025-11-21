use std::sync::Arc;

use chrono::Utc;
use error_stack::Report;
use poem::get;
use poem::listener::TcpListener;
use poem::middleware::Tracing;
use poem::post;
use poem::web::Data;
use poem::web::Json;
use poem::EndpointExt;
use poem::Route;
use poem::Server;
use tokio_util::sync::CancellationToken;
use tracing::error;
use tracing::info;
use trap::http::HttpTrapRequest;
use trap::http::HttpTrapResponse;
use trap::TrapAction;
use trap::TrapError;
use trap::TrapHandler;
use trap::Waker;

use super::auth::JwtAuthMiddleware;
use super::ApiError;
use super::JwtAuthConfig;
use crate::config::AutoFreezeAndResume;
use crate::controller::handlers::get_pod_info;
use crate::controller::handlers::ping;
use crate::controller::handlers::process_init;
use crate::core::pod::traits::{DeviceSnapshotProvider, PodStateRepository, TimeSource};
use crate::core::pod::PodManager;
use crate::platform::limiter_comm::CommandDispatcher;
use utils::shared_memory::traits::SharedMemoryAccess;

/// HTTP API server for querying pod resource information
pub struct ApiServer<M, P, D, T> {
    pod_manager: Arc<PodManager<M, P, D, T>>,
    listen_addr: String,
    jwt_config: JwtAuthConfig,
    trap_handler: Arc<dyn TrapHandler + Send + Sync + 'static>,
    command_dispatcher: Arc<CommandDispatcher>,
    auto_freeze_config: Arc<AutoFreezeAndResume>,
}

impl<M, P, D, T> ApiServer<M, P, D, T>
where
    M: SharedMemoryAccess + 'static,
    P: PodStateRepository + 'static,
    D: DeviceSnapshotProvider + 'static,
    T: TimeSource + 'static,
{
    /// Create a new API server
    pub fn new(
        pod_manager: Arc<PodManager<M, P, D, T>>,
        listen_addr: String,
        jwt_config: JwtAuthConfig,
        trap_handler: Arc<dyn TrapHandler + Send + Sync + 'static>,
        command_dispatcher: Arc<CommandDispatcher>,
        auto_freeze_config: Arc<AutoFreezeAndResume>,
    ) -> Self {
        Self {
            pod_manager,
            listen_addr,
            jwt_config,
            trap_handler,
            command_dispatcher,
            auto_freeze_config,
        }
    }

    /// Start the API server
    ///
    /// # Errors
    ///
    /// - [`ApiError::ServerError`] if the server fails to start or bind to the address
    pub async fn run(self, cancellation_token: CancellationToken) -> Result<(), Report<ApiError>> {
        info!("Starting HTTP API server on {}", self.listen_addr);

        let trap_routes = Route::new().at("/", post(trap_endpoint).data(self.trap_handler.clone()));

        let limiter_routes = self.command_dispatcher.create_routes();

        let app = Route::new()
            // Check endpoints (unprotected)
            .at("/healthz", get(ping))
            .at("/readyz", get(ping))
            // Protected routes with JWT middleware
            .at(
                "/api/v1/pod",
                get(get_pod_info::<M, P, D, T>::default())
                    .with(JwtAuthMiddleware::new(self.jwt_config.clone())),
            )
            .at(
                "/api/v1/process",
                post(process_init::<M, P, D, T>::default())
                    .with(JwtAuthMiddleware::new(self.jwt_config.clone())),
            )
            // Unprotected routes without JWT middleware
            .nest("/api/v1/trap", trap_routes)
            .nest("/api/v1/limiter", limiter_routes)
            .data(self.pod_manager.clone())
            .data(self.command_dispatcher.clone())
            .data(self.auto_freeze_config.clone())
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
            _ = cancellation_token.cancelled() => {
                info!("API server shutdown requested");
                Ok(())
            }
        }
    }
}

/// Simple waker implementation for capturing trap actions
#[derive(Clone)]
struct SimpleWaker {
    action: Arc<tokio::sync::Mutex<Option<TrapAction>>>,
}

impl SimpleWaker {
    fn new() -> Self {
        Self {
            action: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    async fn get_action(&self) -> Option<TrapAction> {
        self.action.lock().await.clone()
    }
}

#[async_trait::async_trait]
impl Waker for SimpleWaker {
    async fn send(&self, _trap_id: u64, action: TrapAction) -> Result<(), TrapError> {
        let action_arc = self.action.clone();
        let mut action_guard = action_arc.lock().await;
        *action_guard = Some(action);
        Ok(())
    }
}

#[poem::handler]
async fn trap_endpoint(
    Json(req): Json<HttpTrapRequest>,
    Data(handler): Data<&Arc<dyn TrapHandler + Send + Sync + 'static>>,
) -> Json<HttpTrapResponse> {
    let waker = SimpleWaker::new();
    handler
        .handle_trap(
            req.process_id,
            req.trap_id.parse::<u64>().unwrap_or(0),
            &req.frame,
            Box::new(waker.clone()),
        )
        .await;
    let action = waker.get_action().await.unwrap_or(TrapAction::Resume);
    Json(HttpTrapResponse {
        trap_id: req.trap_id.clone(),
        action,
        timestamp: Utc::now(),
    })
}
