pub mod ipc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// TrapFrame: context of a trap
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TrapFrame {
    OutOfMemory { requested_bytes: u64 },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TrapAction {
    Resume,
    Fatal(String),
}

#[derive(Error, Debug)]
pub enum TrapError {
    #[error("IPC error: {0}")]
    Ipc(#[from] ipc_channel::Error),
    #[error("IPC recv error: {0}")]
    IpcRecv(#[from] ipc_channel::ipc::IpcError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub trait Trap {
    fn enter_trap_and_wait(&self, frame: TrapFrame) -> Result<TrapAction, TrapError>;
}

pub type Waker = Box<dyn FnOnce(Result<TrapAction, TrapError>) + Send>;

pub trait TrapHandler {
    fn handle_trap(&self, frame: &TrapFrame, waker: Waker);
}
