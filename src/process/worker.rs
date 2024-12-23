use anyhow::{anyhow, Result};
use std::os::unix::net::UnixDatagram;
use std::path::Path;

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
    id: String,
    socket_path: String,
    requested: GpuResources,
}

impl TensorFusionWorker {
    pub fn new(id: String, socket_path: String, requested: GpuResources) -> TensorFusionWorker {
        Self {
            id,
            socket_path,
            requested,
        }
    }

    fn send_message(message_type: ControlMessageType, socket_path: &str) -> Result<()> {
        let message = ControlMessage::new(message_type);
        let socket = UnixDatagram::unbound()?;
        socket.connect(Path::new(socket_path))?;
        let message_bytes = message.as_bytes();

        match socket.send(message_bytes) {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow!("failed to send message: {}", e)),
        }
    }
}

impl GpuProcess for TensorFusionWorker {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn state(&self) -> ProcessState {
        todo!()
    }

    fn requested_resources(&self) -> GpuResources {
        self.requested.clone()
    }

    fn current_resources(&self) -> Result<GpuResources> {
        todo!()
    }

    fn pause(&mut self) -> Result<()> {
        Self::send_message(ControlMessageType::Suspend, &self.socket_path)
    }

    fn release(&mut self) -> Result<()> {
        Self::send_message(ControlMessageType::SuspendAndVramReclaim, &self.socket_path)
    }

    fn resume(&mut self) -> Result<()> {
        Self::send_message(ControlMessageType::Resume, &self.socket_path)
    }
}
