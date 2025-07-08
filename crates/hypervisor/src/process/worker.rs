use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use anyhow::Result;
use api_types::QosLevel;

use super::GpuProcess;
use super::GpuResources;
use super::ProcessState;
use crate::gpu_observer::GpuObserver;

#[allow(dead_code)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ControlMessageType {
    Suspend = 0,
    Resume = 1,
    SuspendAndVramReclaim = 2,
    SuspendAndSave = 3,
    ResponseSuccess = 4,
    ResponseFail = 5,
}

#[repr(C)]
#[derive(Debug)]
struct ControlMessage {
    control: ControlMessageType,
    payload: [u8; 128],
}

impl ControlMessage {
    fn new(control: ControlMessageType) -> Self {
        Self {
            control,
            payload: [0; 128],
        }
    }
}

pub(crate) struct TensorFusionWorker {
    id: u32,
    state: RwLock<ProcessState>,
    gpu_uuids: Vec<String>,
    gpu_observer: Arc<GpuObserver>,
    qos_level: api_types::QosLevel,
    /// Worker name, formatted as "namespace/pod_name"
    pub(crate) name: String,
    /// Kubernetes pod name
    pub(crate) pod_name: String,
    /// Kubernetes namespace
    pub(crate) namespace: String,
}

impl TensorFusionWorker {
    pub(crate) fn new(
        id: u32,
        qos_level: QosLevel,
        gpu_uuids: Vec<String>,
        gpu_observer: Arc<GpuObserver>,
        namespace: String,
        pod_name: String,
    ) -> TensorFusionWorker {
        Self {
            id,
            qos_level,
            state: RwLock::new(ProcessState::Running),
            gpu_uuids,
            gpu_observer,
            name: format!("{}/{}", namespace, pod_name),
            pod_name,
            namespace,
        }
    }

    fn send_message(&self, _message: ControlMessage) -> Result<bool> {
        todo!()
    }
}

impl GpuProcess for TensorFusionWorker {
    fn pid(&self) -> u32 {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn current_resources(&self) -> HashMap<&str, GpuResources> {
        let mut resources: HashMap<&str, GpuResources> = HashMap::new();
        for gpu_uuid in &self.gpu_uuids {
            let resource = self
                .gpu_observer
                .get_process_resources(gpu_uuid, self.pid());
            if let Some(resource) = resource {
                resources.insert(gpu_uuid.as_str(), resource);
            }
        }
        resources
    }

    fn pause(&self) -> Result<()> {
        match self.send_message(ControlMessage::new(ControlMessageType::Suspend)) {
            Ok(true) => {
                // Successfully sent and got positive response
                *self.state.write().expect("poisoned") = ProcessState::Paused;
                Ok(())
            }
            Ok(false) => {
                // Successfully sent but got negative response
                tracing::warn!(
                    "Process ID {} pause request was rejected by the worker",
                    self.id
                );
                Err(anyhow::anyhow!("Pause request was rejected by the worker"))
            }
            Err(e) => {
                // Communication error
                tracing::error!(
                    "Failed to send pause message to process ID {}: {}",
                    self.id,
                    e
                );
                Err(anyhow::anyhow!("Failed to communicate with worker: {}", e))
            }
        }
    }

    fn release(&self) -> Result<()> {
        match self.send_message(ControlMessage::new(
            ControlMessageType::SuspendAndVramReclaim,
        )) {
            Ok(true) => {
                // Successfully sent and got positive response
                *self.state.write().expect("poisoned") = ProcessState::Released;
                Ok(())
            }
            Ok(false) => {
                // Successfully sent but got negative response
                tracing::warn!(
                    "Process ID {} release request was rejected by the worker",
                    self.id
                );
                Err(anyhow::anyhow!(
                    "Release request was rejected by the worker"
                ))
            }
            Err(e) => {
                // Communication error
                tracing::error!(
                    "Failed to send release message to process ID {}: {}",
                    self.id,
                    e
                );
                Err(anyhow::anyhow!("Failed to communicate with worker: {}", e))
            }
        }
    }

    fn resume(&self) -> Result<()> {
        match self.send_message(ControlMessage::new(ControlMessageType::Resume)) {
            Ok(true) => {
                // Successfully sent and got positive response
                *self.state.write().expect("poisoned") = ProcessState::Running;
                Ok(())
            }
            Ok(false) => {
                // Successfully sent but got negative response
                tracing::warn!(
                    "Process ID {} resume request was rejected by the worker",
                    self.id
                );
                Err(anyhow::anyhow!("Resume request was rejected by the worker"))
            }
            Err(e) => {
                // Communication error
                tracing::error!(
                    "Failed to send resume message to process ID {}: {}",
                    self.id,
                    e
                );
                Err(anyhow::anyhow!("Failed to communicate with worker: {}", e))
            }
        }
    }

    fn qos_level(&self) -> super::QosLevel {
        self.qos_level
    }
}

// Custom Clone implementation because some fields are not Clone
impl Clone for TensorFusionWorker {
    fn clone(&self) -> Self {
        // Clone requested fields; reset non-cloneable unix_stream to uninit; clone state value
        let state_value = *self.state.read().expect("poisoned");
        Self {
            id: self.id,
            state: RwLock::new(state_value),
            gpu_uuids: self.gpu_uuids.clone(),
            gpu_observer: Arc::clone(&self.gpu_observer),
            qos_level: self.qos_level,
            name: self.name.clone(),
            pod_name: self.pod_name.clone(),
            namespace: self.namespace.clone(),
        }
    }
}

// Manual Debug implementation to avoid issues with non-Debug fields
impl std::fmt::Debug for TensorFusionWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorFusionWorker")
            .field("id", &self.id)
            .field("state", &*self.state.read().expect("poisoned"))
            .field("gpu_uuids", &self.gpu_uuids)
            .field("qos_level", &self.qos_level)
            .field("name", &self.name)
            .field("pod_name", &self.pod_name)
            .field("namespace", &self.namespace)
            .finish()
    }
}
