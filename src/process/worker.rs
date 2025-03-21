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
        Ok(response.control == ControlMessageType::ResponseSuccess)
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
        self.send_message(ControlMessage::new(ControlMessageType::Suspend))?;
        *self.state.write().expect("poisoned") = ProcessState::Paused;
        Ok(())
    }

    fn release(&self) -> Result<()> {
        self.send_message(ControlMessage::new(
            ControlMessageType::SuspendAndVramReclaim,
        ))?;
        *self.state.write().expect("poisoned") = ProcessState::Released;
        Ok(())
    }

    fn resume(&self) -> Result<()> {
        self.send_message(ControlMessage::new(ControlMessageType::Resume))?;
        *self.state.write().expect("poisoned") = ProcessState::Running;
        Ok(())
    }

    fn gpu_uuid(&self) -> &str {
        &self.gpu_uuid
    }
}
