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
use trap::Waker;

use super::auth::JwtAuthMiddleware;
use super::errors::ApiError;
use super::types::JwtAuthConfig;
use crate::api::handlers::get_worker_info;
use crate::api::handlers::worker_init;
use crate::gpu_observer::GpuObserver;
use crate::limiter_comm::CommandDispatcher;
use crate::worker_manager::WorkerManager;

/// HTTP API server for querying pod resource information
pub struct ApiServer {
    worker_manager: Arc<WorkerManager>,
    listen_addr: String,
    jwt_config: JwtAuthConfig,
    trap_handler: Arc<dyn trap::TrapHandler + Send + Sync + 'static>,
    command_dispatcher: Arc<CommandDispatcher>,
    gpu_observer: Arc<GpuObserver>,
}

impl ApiServer {
    /// Create a new API server
    pub fn new(
        worker_manager: Arc<WorkerManager>,
        listen_addr: String,
        jwt_config: JwtAuthConfig,
        trap_handler: Arc<dyn trap::TrapHandler + Send + Sync + 'static>,
        command_dispatcher: Arc<CommandDispatcher>,
        gpu_observer: Arc<GpuObserver>,
    ) -> Self {
        Self {
            worker_manager,
            listen_addr,
            jwt_config,
            trap_handler,
            command_dispatcher,
            gpu_observer,
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

        // Routes that require JWT authentication
        let protected_routes = Route::new()
            .at("/api/v1/worker", get(get_worker_info))
            .at("/api/v1/worker", post(worker_init))
            .with(JwtAuthMiddleware::new(self.jwt_config));

        // Routes that don't require JWT authentication
        let unprotected_routes = Route::new()
            .nest("/api/v1/trap", trap_routes)
            .nest("/api/v1/limiter", limiter_routes);

        let app = Route::new()
            .nest("/", protected_routes)
            .nest("/", unprotected_routes)
            .data(self.worker_manager.registry().clone())
            .data(self.worker_manager.clone())
            .data(self.command_dispatcher.clone())
            .data(self.gpu_observer.clone())
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
    action: Arc<std::sync::Mutex<Option<TrapAction>>>,
}

impl SimpleWaker {
    fn new() -> Self {
        Self {
            action: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    fn get_action(&self) -> Option<TrapAction> {
        self.action.lock().unwrap().clone()
    }
}

impl Waker for SimpleWaker {
    fn send(&self, _trap_id: u64, action: TrapAction) -> Result<(), TrapError> {
        let mut action_guard = self.action.lock().unwrap();
        *action_guard = Some(action);
        Ok(())
    }
}

#[poem::handler]
async fn trap_endpoint(
    Json(req): Json<HttpTrapRequest>,
    Data(handler): Data<&Arc<dyn trap::TrapHandler + Send + Sync + 'static>>,
) -> Json<HttpTrapResponse> {
    let waker = SimpleWaker::new();
    handler.handle_trap(
        req.process_id,
        req.trap_id.parse::<u64>().unwrap_or(0),
        &req.frame,
        Box::new(waker.clone()),
    );
    let action = waker.get_action().unwrap_or(TrapAction::Resume);
    Json(HttpTrapResponse {
        trap_id: req.trap_id.clone(),
        action,
        timestamp: Utc::now(),
    })
}
