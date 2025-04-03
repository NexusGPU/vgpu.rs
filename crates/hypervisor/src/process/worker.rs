use anyhow::Result;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::{os::unix::net::UnixStream, sync::RwLock};

use crate::gpu_observer::GpuObserver;

use super::{GpuProcess, GpuResources, ProcessState};

#[allow(dead_code)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlMessageType {
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

pub struct TensorFusionWorker {
    id: u32,
    requested: GpuResources,
    state: RwLock<ProcessState>,
    gpu_uuid: String,
    gpu_observer: Arc<GpuObserver>,
    unix_stream: Mutex<UnixStream>,
}

impl TensorFusionWorker {
    pub fn new(
        id: u32,
        socket_path: PathBuf,
        requested: GpuResources,
        gpu_uuid: String,
        gpu_observer: Arc<GpuObserver>,
    ) -> TensorFusionWorker {
        let unix_stream = UnixStream::connect(&socket_path).unwrap();
        Self {
            id,
            unix_stream: Mutex::new(unix_stream),
            requested,
            state: RwLock::new(ProcessState::Running),
            gpu_uuid,
            gpu_observer,
        }
    }

    fn send_message(&self, message: ControlMessage) -> Result<bool> {
        let mut unix_stream = self.unix_stream.lock().unwrap();
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

    fn state(&self) -> ProcessState {
        self.state.read().expect("poisoned").clone()
    }

    fn requested_resources(&self) -> GpuResources {
        self.requested.clone()
    }

    fn current_resources(&self) -> GpuResources {
        self.gpu_observer
            .get_process_resources(&self.gpu_uuid, self.id)
            .unwrap_or(GpuResources {
                memory_bytes: 0,
                compute_percentage: 0,
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

    fn gpu_uuid(&self) -> &str {
        &self.gpu_uuid
    }
}
