pub mod dummy;
#[cfg(feature = "ipc")]
pub mod ipc;

#[cfg(feature = "http")]
pub mod http;

use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;
use thiserror::Error; // Use alias to avoid potential confusion

/// TrapFrame: context of a trap
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TrapFrame {
    OutOfMemory { requested_bytes: u64 },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TrapAction {
    Resume,
    Fatal(String),
}

#[derive(Error, Debug)]
pub enum TrapError {
    #[cfg(feature = "ipc")]
    #[error("IPC error: {0}")]
    Ipc(#[from] ipc_channel::Error),
    #[cfg(feature = "ipc")]
    #[error("IPC recv error: {0}")]
    IpcRecv(#[from] ipc_channel::ipc::IpcError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid IPC message format")]
    InvalidIpcMessage,
}

pub trait Trap {
    fn enter_trap_and_wait(&self, frame: TrapFrame) -> Result<TrapAction, TrapError>;
}

/// Waker trait for sending trap responses
pub trait Waker: Send + Sync {
    fn send(&self, trap_id: u64, action: TrapAction) -> Result<(), TrapError>;
}

pub trait TrapHandler {
    fn handle_trap(&self, pid: u32, trap_id: u64, frame: &TrapFrame, waker: Box<dyn Waker>);
}

impl<T> TrapHandler for Arc<T>
where
    T: TrapHandler + ?Sized,
{
    fn handle_trap(&self, pid: u32, trap_id: u64, frame: &TrapFrame, waker: Box<dyn Waker>) {
        (**self).handle_trap(pid, trap_id, frame, waker);
    }
}
