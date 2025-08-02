use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use anyhow::Result;
use api_types::QosLevel;

use super::GpuProcess;
use super::GpuResources;
use super::ProcessState;
use crate::api::types::LimiterCommandType;
use crate::gpu_observer::GpuObserver;
use crate::limiter_comm::CommandDispatcher;

pub struct TensorFusionWorker {
    id: u32,
    state: RwLock<ProcessState>,
    gpu_uuids: Vec<String>,
    gpu_observer: Arc<GpuObserver>,
    qos_level: QosLevel,
    /// Worker name, formatted as "namespace/pod_name"
    pub(crate) name: String,
    /// Kubernetes pod name
    pub(crate) pod_name: String,
    /// Kubernetes namespace
    pub(crate) namespace: String,
    /// Command dispatcher for sending commands to limiters
    command_dispatcher: Arc<CommandDispatcher>,
}

impl TensorFusionWorker {
    pub(crate) fn new(
        id: u32,
        qos_level: QosLevel,
        gpu_uuids: Vec<String>,
        gpu_observer: Arc<GpuObserver>,
        namespace: String,
        pod_name: String,
        command_dispatcher: Arc<CommandDispatcher>,
    ) -> TensorFusionWorker {
        Self {
            id,
            qos_level,
            state: RwLock::new(ProcessState::Running),
            gpu_uuids,
            gpu_observer,
            name: format!("{namespace}-{pod_name}"),
            pod_name,
            namespace,
            command_dispatcher,
        }
    }

    /// Generate limiter ID based on worker information
    fn get_limiter_id(&self) -> String {
        format!("limiter_{}", self.id)
    }

    /// Send command to limiter using CommandDispatcher
    fn send_command(&self, command_type: LimiterCommandType) -> Result<()> {
        let limiter_id = self.get_limiter_id();
        let command_dispatcher = Arc::clone(&self.command_dispatcher);
        let process_id = self.id;

        // Clone values for use in the async closure
        let limiter_id_clone = limiter_id.clone();
        let command_type_clone = command_type.clone();

        // Spawn a task to send the command asynchronously without blocking
        tokio::spawn(async move {
            match command_dispatcher
                .enqueue_command(&limiter_id_clone, command_type_clone.clone())
                .await
            {
                Ok(_command_id) => {
                    tracing::info!(
                        "Command {:?} sent successfully to limiter {} for process {}",
                        command_type_clone,
                        limiter_id_clone,
                        process_id
                    );
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to send command {:?} to limiter {} for process {}: {}",
                        command_type_clone,
                        limiter_id_clone,
                        process_id,
                        e
                    );
                }
            }
        });

        // Return immediately - the command will be sent asynchronously
        tracing::info!(
            "Queued command {:?} for limiter {} for process {}",
            command_type,
            limiter_id,
            self.id
        );
        Ok(())
    }
}

#[async_trait::async_trait]
impl GpuProcess for TensorFusionWorker {
    fn pid(&self) -> u32 {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn current_resources(&self) -> HashMap<&str, GpuResources> {
        let mut resources: HashMap<&str, GpuResources> = HashMap::new();
        for gpu_uuid in &self.gpu_uuids {
            let resource = self
                .gpu_observer
                .get_process_resources(gpu_uuid, self.pid())
                .await;
            if let Some(resource) = resource {
                resources.insert(gpu_uuid.as_str(), resource);
            }
        }
        resources
    }

    async fn pause(&self) -> Result<()> {
        match self.send_command(LimiterCommandType::TfSuspend) {
            Ok(()) => {
                *self.state.write().await = ProcessState::Paused;
                tracing::info!("Process {} paused successfully", self.id);
                Ok(())
            }
            Err(e) => {
                tracing::error!("Failed to pause process {}: {}", self.id, e);
                Err(e)
            }
        }
    }

    async fn release(&self) -> Result<()> {
        match self.send_command(LimiterCommandType::TfVramReclaim) {
            Ok(()) => {
                *self.state.write().await = ProcessState::Released;
                tracing::info!("Process {} released successfully", self.id);
                Ok(())
            }
            Err(e) => {
                tracing::error!("Failed to release process {}: {}", self.id, e);
                Err(e)
            }
        }
    }

    async fn resume(&self) -> Result<()> {
        match self.send_command(LimiterCommandType::TfResume) {
            Ok(()) => {
                *self.state.write().await = ProcessState::Running;
                tracing::info!("Process {} resumed successfully", self.id);
                Ok(())
            }
            Err(e) => {
                tracing::error!("Failed to resume process {}: {}", self.id, e);
                Err(e)
            }
        }
    }

    fn qos_level(&self) -> super::QosLevel {
        self.qos_level
    }
}

// Manual Debug implementation to avoid issues with non-Debug fields
impl std::fmt::Debug for TensorFusionWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // For Debug, we can't use async, so we show a placeholder for state
        f.debug_struct("TensorFusionWorker")
            .field("id", &self.id)
            .field("gpu_uuids", &self.gpu_uuids)
            .field("qos_level", &self.qos_level)
            .field("name", &self.name)
            .field("pod_name", &self.pod_name)
            .field("namespace", &self.namespace)
            .finish()
    }
}
