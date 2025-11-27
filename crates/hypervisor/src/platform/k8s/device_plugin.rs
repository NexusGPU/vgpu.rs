#![expect(clippy::doc_markdown, reason = "Generated proto code")]
pub mod api {
    tonic::include_proto!("v1beta1");
}

use core::fmt;
use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use api::device_plugin_server::DevicePlugin;
use api::device_plugin_server::DevicePluginServer;
use api::registration_client::RegistrationClient;
use api::AllocateRequest;
use api::AllocateResponse;
use api::ContainerAllocateResponse;
use api::DevicePluginOptions;
use api::DeviceSpec;
use api::Empty;
use api::ListAndWatchResponse;
use api::PreStartContainerRequest;
use api::PreStartContainerResponse;
use api::PreferredAllocationRequest;
use api::PreferredAllocationResponse;
use api::RegisterRequest;
use error_stack::Report;
use error_stack::ResultExt as _;
use futures::Stream;
use hyper_util::rt::TokioIo;
use nvml_wrapper::Nvml;
use tokio::net::UnixListener;
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;
use tonic::transport::Endpoint;
use tonic::transport::Uri;
use tonic::Request;
use tonic::Response;
use tonic::Result as TonicResult;
use tonic::Status;
use tower::service_fn;
use tracing::debug;
use tracing::error;
use tracing::info;

use crate::platform::nvml::init_nvml;

#[derive(Debug)]
pub enum DevicePluginError {
    SocketCleanup,
    SocketBind,
    UdsConnection,
    Registration,
}

impl fmt::Display for DevicePluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SocketCleanup => write!(f, "failed to clean up old socket file"),
            Self::SocketBind => write!(f, "failed to bind Unix socket"),
            Self::UdsConnection => write!(f, "failed to create UDS connection"),
            Self::Registration => write!(f, "failed to register with kubelet"),
        }
    }
}

impl core::error::Error for DevicePluginError {}

/// GPU Device Plugin for Kubernetes
#[derive(Debug)]
pub struct GpuDevicePlugin {
    /// device plugin endpoint name
    endpoint: String,
    /// resource name (e.g. "tensor-fusion.ai/index")
    resource_name: String,
    /// device plugin options
    options: DevicePluginOptions,
}

impl GpuDevicePlugin {
    /// Create a new GPU Device Plugin instance
    pub fn new(endpoint: String, resource_name: String) -> Arc<Self> {
        let options = DevicePluginOptions {
            pre_start_required: false,
            get_preferred_allocation_available: false,
        };

        Arc::new(Self {
            endpoint,
            resource_name,
            options,
        })
    }

    /// Start device plugin server
    ///
    /// Returns a oneshot receiver that will be signaled when the server is ready
    pub async fn start(
        self: &Arc<Self>,
        socket_path: impl AsRef<Path>,
        cancellation_token: CancellationToken,
    ) -> Result<oneshot::Receiver<()>, Report<DevicePluginError>> {
        let socket_path = socket_path.as_ref();
        info!("start device plugin server: {}", socket_path.display());

        // clean up old socket file if it exists
        if socket_path.exists() {
            tokio::fs::remove_file(socket_path)
                .await
                .change_context(DevicePluginError::SocketCleanup)?;
        }

        // create Unix socket listener
        let listener =
            UnixListener::bind(socket_path).change_context(DevicePluginError::SocketBind)?;

        // create DevicePlugin service
        let device_plugin_service =
            DevicePluginService::new(self.clone(), cancellation_token.clone());
        let device_plugin_server = DevicePluginServer::new(device_plugin_service);

        info!("gRPC server is bound to: {}", socket_path.display());

        // create ready signal channel
        let (ready_tx, ready_rx) = oneshot::channel();

        // start gRPC server
        tokio::spawn(async move {
            // signal that server is ready before starting to serve
            let _ = ready_tx.send(());

            if let Err(e) = tonic::transport::Server::builder()
                .add_service(device_plugin_server)
                .serve_with_incoming_shutdown(
                    tokio_stream::wrappers::UnixListenerStream::new(listener),
                    async move {
                        cancellation_token.cancelled().await;
                        info!("shutting down gRPC server");
                    },
                )
                .await
            {
                error!("gRPC server error: {e}");
            }
        });

        Ok(ready_rx)
    }

    /// Register device plugin with kubelet
    pub async fn register_with_kubelet(
        &self,
        kubelet_socket: impl AsRef<Path>,
    ) -> Result<(), Report<DevicePluginError>> {
        let kubelet_socket = kubelet_socket.as_ref();
        info!(
            "registering device plugin with kubelet: {}",
            kubelet_socket.display()
        );

        // create UDS client connection
        let channel = self
            .create_uds_channel(kubelet_socket)
            .await
            .change_context(DevicePluginError::UdsConnection)?;
        let mut client = RegistrationClient::new(channel);

        // create registration request
        let request = RegisterRequest {
            version: "v1beta1".to_string(),
            endpoint: self.endpoint.clone(),
            resource_name: self.resource_name.clone(),
            options: Some(self.options),
        };

        // send registration request
        client
            .register(Request::new(request))
            .await
            .change_context(DevicePluginError::Registration)?;

        info!("successfully registered device plugin with kubelet");
        Ok(())
    }

    /// Create Unix Domain Socket client connection
    async fn create_uds_channel(
        &self,
        socket_path: impl AsRef<Path>,
    ) -> Result<Channel, tonic::transport::Error> {
        let socket_path = socket_path.as_ref().to_path_buf();

        // create UDS connection using TokioIo wrapper
        // Note: The HTTP URL is a placeholder since we're using Unix socket connector
        Endpoint::from_static("http://tonic")
            .connect_with_connector(service_fn(move |_: Uri| {
                let socket_path = socket_path.clone();
                async move {
                    match UnixStream::connect(socket_path).await {
                        Ok(stream) => Ok(TokioIo::new(stream)),
                        Err(e) => Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                    }
                }
            }))
            .await
    }
}

/// DevicePlugin service implementation
/// provide core device management functionality
#[derive(Debug)]
pub struct DevicePluginService {
    device_plugin: Arc<GpuDevicePlugin>,
    cancellation_token: CancellationToken,
}

impl DevicePluginService {
    pub fn new(device_plugin: Arc<GpuDevicePlugin>, cancellation_token: CancellationToken) -> Self {
        Self {
            device_plugin,
            cancellation_token,
        }
    }

    /// Collect all NVIDIA devices that should be passed to containers
    fn get_nvidia_devices(nvml: &Nvml) -> Vec<DeviceSpec> {
        let mut devices = Vec::new();

        // add control device
        devices.push(DeviceSpec {
            container_path: "/dev/nvidiactl".to_string(),
            host_path: "/dev/nvidiactl".to_string(),
            permissions: "rwm".to_string(),
        });

        // add UVM devices
        devices.push(DeviceSpec {
            container_path: "/dev/nvidia-uvm".to_string(),
            host_path: "/dev/nvidia-uvm".to_string(),
            permissions: "rwm".to_string(),
        });

        devices.push(DeviceSpec {
            container_path: "/dev/nvidia-uvm-tools".to_string(),
            host_path: "/dev/nvidia-uvm-tools".to_string(),
            permissions: "rwm".to_string(),
        });

        // add GPU devices based on NVML device count
        let device_count = nvml.device_count().unwrap_or(0);
        for i in 0..device_count {
            let device_path = format!("/dev/nvidia{i}");
            devices.push(DeviceSpec {
                container_path: device_path.clone(),
                host_path: device_path,
                permissions: "rwm".to_string(),
            });
        }

        devices
    }
}

#[tonic::async_trait]
impl DevicePlugin for DevicePluginService {
    async fn get_device_plugin_options(
        &self,
        _request: Request<Empty>,
    ) -> TonicResult<Response<DevicePluginOptions>> {
        debug!("getting device plugin options");
        Ok(Response::new(self.device_plugin.options))
    }

    type ListAndWatchStream =
        Pin<Box<dyn Stream<Item = Result<ListAndWatchResponse, Status>> + Send>>;

    async fn list_and_watch(
        &self,
        _request: Request<Empty>,
    ) -> TonicResult<Response<Self::ListAndWatchStream>> {
        info!("starting to watch device list");

        let (tx, rx) = mpsc::unbounded_channel();
        let cancellation_token = self.cancellation_token.clone();

        tokio::spawn(async move {
            // create devices with IDs from 0 to 255
            let devices: Vec<api::Device> = (0..=255)
                .map(|i| api::Device {
                    id: i.to_string(),
                    health: "Healthy".to_string(),
                    topology: None,
                })
                .collect();

            // send initial device list
            let initial_response = ListAndWatchResponse { devices };

            if let Err(e) = tx.send(Ok(initial_response)) {
                error!("failed to send initial device list: {e}");
                return;
            }
            // listen for cancellation signal
            cancellation_token.cancelled().await;
            info!("device watch task stopped");
        });

        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_preferred_allocation(
        &self,
        request: Request<PreferredAllocationRequest>,
    ) -> TonicResult<Response<PreferredAllocationResponse>> {
        let req = request.into_inner();
        debug!("getting preferred device allocation: {:?}", req);
        let response = PreferredAllocationResponse {
            container_responses: vec![],
        };
        Ok(Response::new(response))
    }

    async fn allocate(
        &self,
        request: Request<AllocateRequest>,
    ) -> TonicResult<Response<AllocateResponse>> {
        let req = request.into_inner();
        info!("allocating devices to container: {:?}", req);

        // initialize NVML for device discovery
        let nvml = match init_nvml() {
            Ok(nvml) => nvml,
            Err(e) => {
                error!("failed to initialize NVML: {e}");
                return Err(Status::internal(format!("failed to initialize NVML: {e}")));
            }
        };

        let mut container_responses = Vec::new();

        for container_req in req.container_requests {
            info!(
                "allocating devices to container, device IDs: {:?}",
                container_req.devices_ids
            );

            // collect all NVIDIA devices to be passed to the container
            let devices = Self::get_nvidia_devices(&nvml);

            let container_response = ContainerAllocateResponse {
                envs: HashMap::new(),
                mounts: Vec::new(),
                devices,
                annotations: HashMap::new(),
                cdi_devices: Vec::new(),
            };

            container_responses.push(container_response);
        }

        let response = AllocateResponse {
            container_responses,
        };

        info!(
            "device allocation completed, allocated {} NVIDIA devices",
            response
                .container_responses
                .first()
                .map(|r| r.devices.len())
                .unwrap_or(0)
        );
        Ok(Response::new(response))
    }

    async fn pre_start_container(
        &self,
        _request: Request<PreStartContainerRequest>,
    ) -> TonicResult<Response<PreStartContainerResponse>> {
        // pre_start_required is set to false, this should not be called
        // but we need to implement it for the trait
        Ok(Response::new(PreStartContainerResponse {}))
    }
}
