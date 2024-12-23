use anyhow::{anyhow, Result};
use std::path::Path;
use std::{os::unix::net::UnixDatagram, sync::RwLock};

use super::{GpuProcess, GpuResources, ProcessState};
use nvml_wrapper::Device;

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

pub struct TensorFusionWorker<'nvml> {
    id: u32,
    socket_path: String,
    requested: GpuResources,
    state: RwLock<ProcessState>,
    device: Device<'nvml>,
}

impl<'nvml> TensorFusionWorker<'nvml> {
    pub fn new(
        id: u32,
        socket_path: String,
        requested: GpuResources,
        device: Device<'nvml>,
    ) -> TensorFusionWorker<'nvml> {
        Self {
            id,
            socket_path,
            requested,
            state: RwLock::new(ProcessState::Running),
            device,
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

impl<'nvml> GpuProcess for TensorFusionWorker<'nvml> {
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
        // Get memory info
        let memory_info = self.device.memory_info()?;
        // Get utilization rates
        let utilization = self.device.utilization_rates()?;

        Ok(GpuResources {
            memory_bytes: memory_info.used,
            compute_percentage: utilization.gpu as u32,
        })
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
}
