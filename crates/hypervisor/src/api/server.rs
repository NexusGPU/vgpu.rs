use error_stack::Report;
use poem::get;
use poem::listener::TcpListener;
use poem::middleware::Tracing;
use poem::EndpointExt;
use poem::Route;
use poem::Server;
use tokio::sync::oneshot;
use tracing::error;
use tracing::info;

use super::auth::JwtAuthMiddleware;
use super::errors::ApiError;
use super::handlers::get_pod_info;
use super::storage::PodStorage;
use super::types::JwtAuthConfig;

/// HTTP API server for querying pod resource information
pub struct ApiServer {
    pod_storage: PodStorage,
    listen_addr: String,
    jwt_config: JwtAuthConfig,
}

impl ApiServer {
    /// Create a new API server
    pub fn new(pod_storage: PodStorage, listen_addr: String, jwt_config: JwtAuthConfig) -> Self {
        Self {
            pod_storage,
            listen_addr,
            jwt_config,
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
            .at("/api/v1/pods", get(get_pod_info))
            .data(self.pod_storage)
            .with(JwtAuthMiddleware::new(self.jwt_config))
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::RwLock;

    use super::*;

    fn create_test_storage() -> PodStorage {
        Arc::new(RwLock::new(HashMap::new()))
    }

    fn create_test_jwt_config() -> JwtAuthConfig {
        JwtAuthConfig {
            public_key: "test-public-key".to_string(),
        }
    }

    #[test]
    fn api_server_can_be_created() {
        // Arrange
        let pod_storage = create_test_storage();
        let listen_addr = "127.0.0.1:8080".to_string();
        let jwt_config = create_test_jwt_config();

        // Act
        let server = ApiServer::new(pod_storage.clone(), listen_addr.clone(), jwt_config.clone());

        // Assert
        assert_eq!(
            server.listen_addr, listen_addr,
            "Listen address should be set correctly"
        );
        assert_eq!(
            server.jwt_config.public_key, "test-public-key",
            "JWT config should be set correctly"
        );
        // Note: We can't directly compare Arc<RwLock<HashMap<...>>> but we can verify it's the same pointer
        assert!(
            Arc::ptr_eq(&server.pod_storage, &pod_storage),
            "Pod storage should be the same reference"
        );
    }

    #[test]
    fn api_server_handles_different_listen_addresses() {
        // Arrange
        let pod_storage = create_test_storage();
        let jwt_config = create_test_jwt_config();

        let test_addresses = vec![
            "0.0.0.0:8080",
            "127.0.0.1:3000",
            "localhost:9090",
            "[::1]:8080", // IPv6
        ];

        for addr in test_addresses {
            // Act
            let server = ApiServer::new(pod_storage.clone(), addr.to_string(), jwt_config.clone());

            // Assert
            assert_eq!(
                server.listen_addr, addr,
                "Server should handle address: {addr}"
            );
        }
    }

    #[test]
    fn api_server_stores_jwt_config_correctly() {
        // Arrange
        let pod_storage = create_test_storage();
        let listen_addr = "127.0.0.1:8080".to_string();

        let jwt_configs = vec![
            JwtAuthConfig {
                public_key: "key1".to_string(),
            },
            JwtAuthConfig {
                public_key: "very-long-key-12345".to_string(),
            },
            JwtAuthConfig {
                public_key: "".to_string(),
            }, // Empty key
        ];

        for jwt_config in jwt_configs {
            // Act
            let server =
                ApiServer::new(pod_storage.clone(), listen_addr.clone(), jwt_config.clone());

            // Assert
            assert_eq!(
                server.jwt_config.public_key, jwt_config.public_key,
                "JWT config should be stored correctly"
            );
        }
    }

    #[tokio::test]
    async fn api_server_graceful_shutdown_setup() {
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        // Act - Immediately send shutdown signal
        shutdown_tx.send(()).expect("should send shutdown signal");

        // Assert
        // Consume the shutdown receiver to verify it works
        let shutdown_result = shutdown_rx.await;
        assert!(
            shutdown_result.is_ok(),
            "Shutdown receiver should work correctly"
        );
    }

    #[test]
    fn api_server_with_minimal_configuration() {
        // Arrange
        let pod_storage = create_test_storage();
        let listen_addr = "127.0.0.1:8080".to_string();
        let jwt_config = JwtAuthConfig {
            public_key: String::new(), // Minimal config with empty key
        };

        // Act
        let server = ApiServer::new(pod_storage, listen_addr.clone(), jwt_config);

        // Assert
        assert_eq!(
            server.listen_addr, listen_addr,
            "Should accept minimal configuration"
        );
        assert_eq!(
            server.jwt_config.public_key, "",
            "Should handle empty JWT key"
        );
    }
}
