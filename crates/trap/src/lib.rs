pub mod dummy;
#[cfg(feature = "ipc")]
pub mod ipc;

#[cfg(feature = "http")]
pub mod http;

use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

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
#[async_trait::async_trait]
pub trait Waker: Send + Sync {
    async fn send(&self, trap_id: u64, action: TrapAction) -> Result<(), TrapError>;
}

#[async_trait::async_trait]
pub trait TrapHandler {
    async fn handle_trap(&self, pid: u32, trap_id: u64, frame: &TrapFrame, waker: Box<dyn Waker>);
}

#[async_trait::async_trait]
impl<T> TrapHandler for Arc<T>
where
    T: TrapHandler + Send + Sync + ?Sized,
{
    async fn handle_trap(&self, pid: u32, trap_id: u64, frame: &TrapFrame, waker: Box<dyn Waker>) {
        TrapHandler::handle_trap(&**self, pid, trap_id, frame, waker).await;
    }
}
