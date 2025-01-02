use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{os::unix::net::UnixDatagram, sync::RwLock};

use crate::gpu_observer::GpuObserver;

use super::{GpuProcess, GpuResources, ProcessState};

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
enum ControlMessageType {
    Suspend = 0,
    SuspendAndVramReclaim = 1,
    Resume = 2,
}

#[repr(C, packed)]
#[derive(Debug)]
struct ControlMessage {
    control_type: u8,
    payload: [u8; 128],
}

impl ControlMessage {
    fn new(control: ControlMessageType) -> Self {
        Self {
            control_type: control as u8,
            payload: [0; 128],
        }
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const ControlMessage) as *const u8,
                std::mem::size_of::<ControlMessage>(),
            )
        }
    }
}

pub struct TensorFusionWorker {
    id: u32,
    socket_path: PathBuf,
    requested: GpuResources,
    state: RwLock<ProcessState>,
    gpu_uuid: String,
    gpu_observer: Arc<GpuObserver>,
}

impl TensorFusionWorker {
    pub fn new(
        id: u32,
        socket_path: PathBuf,
        requested: GpuResources,
        gpu_uuid: String,
        gpu_observer: Arc<GpuObserver>,
    ) -> TensorFusionWorker {
        Self {
            id,
            socket_path,
            requested,
            state: RwLock::new(ProcessState::Running),
            gpu_uuid,
            gpu_observer,
        }
    }

    fn send_message(message_type: ControlMessageType, socket_path: &Path) -> Result<()> {
        let message = ControlMessage::new(message_type);
        let socket = UnixDatagram::unbound()?;
        socket.connect(socket_path)?;
        let message_bytes = message.as_bytes();

        match socket.send(message_bytes) {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow!("failed to send message: {}", e)),
        }
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

    fn current_resources(&self) -> Result<GpuResources> {
        self.gpu_observer
            .get_process_resources(&self.gpu_uuid, self.id)
            .ok_or_else(|| anyhow!("Process resources not found"))
    }

    fn pause(&self) -> Result<()> {
        Self::send_message(ControlMessageType::Suspend, &self.socket_path)?;
        *self.state.write().expect("poisoned") = ProcessState::Paused;
        Ok(())
    }

    fn release(&self) -> Result<()> {
        Self::send_message(ControlMessageType::SuspendAndVramReclaim, &self.socket_path)?;
        *self.state.write().expect("poisoned") = ProcessState::Released;
        Ok(())
    }

    fn resume(&self) -> Result<()> {
        Self::send_message(ControlMessageType::Resume, &self.socket_path)?;
        *self.state.write().expect("poisoned") = ProcessState::Running;
        Ok(())
    }

    fn gpu_uuid(&self) -> &str {
        &self.gpu_uuid
    }
}
