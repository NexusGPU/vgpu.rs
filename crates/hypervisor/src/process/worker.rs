use std::io::Read;
use std::io::Write;
use std::mem::MaybeUninit;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;

use anyhow::Result;

use super::GpuProcess;
use super::GpuResources;
use super::ProcessState;
use super::QosLevel;
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
    #[allow(dead_code)]
    requested: GpuResources,
    socket_path: PathBuf,
    state: RwLock<ProcessState>,
    gpu_uuid: String,
    gpu_observer: Arc<GpuObserver>,
    unix_stream: MaybeUninit<Mutex<UnixStream>>,
    qos_level: QosLevel,
    /// Kubernetes pod name (if applicable)
    #[allow(dead_code)]
    pub(crate) pod_name: Option<String>,
    /// Kubernetes namespace (if applicable)
    #[allow(dead_code)]
    pub(crate) namespace: Option<String>,
    /// Kubernetes UID for tracking pod lifecycle
    #[allow(dead_code)]
    pub(crate) kubernetes_uid: Option<String>,
}

impl TensorFusionWorker {
    pub(crate) fn new(
        id: u32,
        socket_path: PathBuf,
        requested: GpuResources,
        qos_level: QosLevel,
        gpu_uuid: String,
        gpu_observer: Arc<GpuObserver>,
    ) -> TensorFusionWorker {
        Self {
            id,
            socket_path,
            unix_stream: MaybeUninit::uninit(),
            qos_level,
            requested,
            state: RwLock::new(ProcessState::Running),
            gpu_uuid,
            gpu_observer,
            pod_name: None,
            namespace: None,
            kubernetes_uid: None,
        }
    }

    pub(crate) fn connect(&mut self) -> Result<()> {
        let unix_stream = UnixStream::connect(&self.socket_path)?;

        self.unix_stream = MaybeUninit::new(Mutex::new(unix_stream));
        Ok(())
    }

    /// Update Kubernetes-related information for this worker.
    #[allow(dead_code)]
    pub(crate) fn update_kubernetes_info(
        &mut self,
        pod_name: Option<String>,
        namespace: Option<String>,
        kubernetes_uid: Option<String>,
    ) {
        self.pod_name = pod_name;
        self.namespace = namespace;
        self.kubernetes_uid = kubernetes_uid;
    }

    /// Update resource requirements from Kubernetes annotations.
    #[allow(dead_code)]
    pub(crate) fn update_resources_from_annotations(
        &mut self,
        annotations: &crate::k8s::TensorFusionAnnotations,
    ) {
        self.requested.tflops_request = annotations.tflops_request;
        self.requested.tflops_limit = annotations.tflops_limit;
        self.requested.memory_limit = annotations.vram_limit;

        // If annotations specify VRAM request, update memory_bytes
        if let Some(vram_request) = annotations.vram_request {
            self.requested.memory_bytes = vram_request;
        }
    }

    fn send_message(&self, message: ControlMessage) -> Result<bool> {
        let mut unix_stream = unsafe { self.unix_stream.assume_init_ref() }
            .lock()
            .unwrap();
        // Send the message
        let message_bytes = unsafe {
            std::slice::from_raw_parts(
                &message as *const ControlMessage as *const u8,
                std::mem::size_of::<ControlMessage>(),
            )
        };
        unix_stream.write_all(message_bytes)?;

        // Read response
        let mut response = ControlMessage::new(ControlMessageType::ResponseSuccess);
        let response_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                &mut response as *mut ControlMessage as *mut u8,
                std::mem::size_of::<ControlMessage>(),
            )
        };
        unix_stream.read_exact(response_bytes)?;
        let succ = response.control == ControlMessageType::ResponseSuccess;
        if !succ {
            tracing::error!(
                "Failed to send control message, control: {:?}",
                response.control
            );
        }
        Ok(succ)
    }
}

impl GpuProcess for TensorFusionWorker {
    fn id(&self) -> u32 {
        self.id
    }

    // fn state(&self) -> ProcessState {
    //     *self.state.read().expect("poisoned")
    // }

    fn requested_resources(&self) -> GpuResources {
        self.requested.clone()
    }

    fn current_resources(&self) -> GpuResources {
        self.gpu_observer
            .get_process_resources(&self.gpu_uuid, self.id)
            .unwrap_or(GpuResources {
                memory_bytes: 0,
                compute_percentage: 0,
                tflops_request: None,
                tflops_limit: None,
                memory_limit: None,
            })
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

    // fn gpu_uuid(&self) -> &str {
    //     &self.gpu_uuid
    // }

    fn qos_level(&self) -> super::QosLevel {
        self.qos_level
    }
}
