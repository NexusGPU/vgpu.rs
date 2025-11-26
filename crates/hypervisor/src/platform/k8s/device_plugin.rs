#![allow(clippy::doc_markdown)]
pub mod api {
    #![allow(clippy::doc_overindented_list_items)]
    tonic::include_proto!("v1beta1");
}

use std::collections::HashMap;
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
use futures::Stream;
use hyper_util::rt::TokioIo;
use tokio::net::UnixListener;
use tokio::net::UnixStream;
use tokio::sync::mpsc;
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

/// GPU Device Plugin for Kubernetes
#[derive(Debug)]
pub struct GpuDevicePlugin {
    /// device plugin endpoint name
    endpoint: String,
    /// resource name (e.g. "tensor-fusion.ai/index")
    resource_name: String,
    /// host path to mount
    host_path: String,
    /// device plugin options
    options: DevicePluginOptions,
}

impl GpuDevicePlugin {
    /// create a new GPU Device Plugin instance
    pub fn new(
        endpoint: String,
        resource_name: String,
        host_path: String,
        pre_start_required: bool,
        get_preferred_allocation_available: bool,
    ) -> Arc<Self> {
        let options = DevicePluginOptions {
            pre_start_required,
            get_preferred_allocation_available,
        };

        Arc::new(Self {
            endpoint,
            resource_name,
            host_path,
            options,
        })
    }

    /// start device plugin server
    pub async fn start(
        self: &Arc<Self>,
        socket_path: &str,
        cancellation_token: CancellationToken,
    ) -> anyhow::Result<()> {
        info!("start device plugin server: {}", socket_path);

        // clean up old socket file if it exists
        if std::path::Path::new(socket_path).exists() {
            std::fs::remove_file(socket_path)?;
        }

        // create Unix socket listener
        let listener = UnixListener::bind(socket_path)?;

        // create DevicePlugin service
        let device_plugin_service =
            DevicePluginService::new(self.clone(), cancellation_token.clone());
        let device_plugin_server = DevicePluginServer::new(device_plugin_service);

        info!("gRPC server is bound to: {}", socket_path);

        // start gRPC server
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(device_plugin_server)
                .serve_with_incoming_shutdown(
                    tokio_stream::wrappers::UnixListenerStream::new(listener),
                    async move {
                        cancellation_token.cancelled().await;
                        info!("shutting down gRPC server");
                    },
                )
                .await
        });

        Ok(())
    }

    /// register device plugin with kubelet
    pub async fn register_with_kubelet(&self, kubelet_socket: &str) -> anyhow::Result<()> {
        info!("registering device plugin with kubelet: {}", kubelet_socket);

        // create UDS client connection
        let channel = self.create_uds_channel(kubelet_socket).await?;
        let mut client = RegistrationClient::new(channel);

        // create registration request
        let request = RegisterRequest {
            version: "v1beta1".to_string(),
            endpoint: self.endpoint.clone(),
            resource_name: self.resource_name.clone(),
            options: Some(self.options),
        };

        // send registration request
        match client.register(Request::new(request)).await {
            Ok(_) => {
                info!("successfully registered device plugin with kubelet");
                Ok(())
            }
            Err(e) => Err(anyhow::anyhow!("registration failed: {e}")),
        }
    }

    /// create Unix Domain Socket client connection
    async fn create_uds_channel(&self, socket_path: &str) -> anyhow::Result<Channel> {
        let socket_path = socket_path.to_string();

        // create UDS connection using TokioIo wrapper
        // Note: The HTTP URL is a placeholder since we're using Unix socket connector
        let channel = Endpoint::from_static("http://tonic")
            .connect_with_connector(service_fn(move |_: Uri| {
                let socket_path = socket_path.clone();
                async move {
                    match UnixStream::connect(socket_path).await {
                        Ok(stream) => Ok(TokioIo::new(stream)),
                        Err(e) => Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                    }
                }
            }))
            .await?;

        Ok(channel)
    }
}

/// DevicePlugin service implementation
/// provide core device management functionality
#[derive(Debug)]
pub struct DevicePluginService {
    device_plugin: Arc<GpuDevicePlugin>,
    /// device state change notification sender
    cancellation_token: CancellationToken,
}

impl DevicePluginService {
    pub fn new(device_plugin: Arc<GpuDevicePlugin>, cancellation_token: CancellationToken) -> Self {
        Self {
            device_plugin,
            cancellation_token,
        }
    }

    /// collect all NVIDIA devices that should be passed to containers
    fn get_nvidia_devices() -> Vec<DeviceSpec> {
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

        // add all GPU devices (nvidia0 to nvidia255)
        // we expose all possible GPUs, kubelet will handle the filtering
        for i in 0..=255 {
            let device_path = format!("/dev/nvidia{i}");
            if std::path::Path::new(&device_path).exists() {
                devices.push(DeviceSpec {
                    container_path: device_path.clone(),
                    host_path: device_path,
                    permissions: "rwm".to_string(),
                });
            }
        }

        devices
    }
}

#[tonic::async_trait]
impl DevicePlugin for DevicePluginService {
    /// get device plugin options
    async fn get_device_plugin_options(
        &self,
        _request: Request<Empty>,
    ) -> TonicResult<Response<DevicePluginOptions>> {
        debug!("getting device plugin options");

        Ok(Response::new(self.device_plugin.options))
    }

    type ListAndWatchStream =
        Pin<Box<dyn Stream<Item = Result<ListAndWatchResponse, Status>> + Send>>;

    /// list and watch device state changes
    async fn list_and_watch(
        &self,
        _request: Request<Empty>,
    ) -> TonicResult<Response<Self::ListAndWatchStream>> {
        info!("starting to watch device list");

        let (tx, rx) = mpsc::unbounded_channel();
        let cancellation_token = self.cancellation_token.clone();

        // start device watch task
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
                error!("failed to send initial device list: {}", e);
                return;
            }
            // listen for cancellation signal
            cancellation_token.cancelled().await;
            info!("device watch task stopped");
        });

        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    /// get preferred device allocation
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

    /// allocate devices to container
    async fn allocate(
        &self,
        request: Request<AllocateRequest>,
    ) -> TonicResult<Response<AllocateResponse>> {
        let req = request.into_inner();
        info!("allocating devices to container: {:?}", req);

        let mut container_responses = Vec::new();

        for container_req in req.container_requests {
            info!(
                "allocating devices to container, device IDs: {:?}",
                container_req.devices_ids
            );

            // collect all NVIDIA devices to be passed to the container
            let devices = Self::get_nvidia_devices();

            let envs = HashMap::new();
            let container_response = ContainerAllocateResponse {
                envs,
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

    /// pre-start container
    async fn pre_start_container(
        &self,
        request: Request<PreStartContainerRequest>,
    ) -> TonicResult<Response<PreStartContainerResponse>> {
        let req = request.into_inner();
        debug!("pre-start container processing: {:?}", req);

        // simple pre-start processing, ensure host path exists
        if !std::path::Path::new(&self.device_plugin.host_path).exists() {
            error!("host path does not exist: {}", self.device_plugin.host_path);
            return Err(Status::internal(format!(
                "host path does not exist: {}",
                self.device_plugin.host_path
            )));
        }

        info!(
            "pre-start check completed, host path exists: {}",
            self.device_plugin.host_path
        );
        Ok(Response::new(PreStartContainerResponse {}))
    }
}
