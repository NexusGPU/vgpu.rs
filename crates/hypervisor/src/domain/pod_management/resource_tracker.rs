//! Process resource tracking module

use std::sync::Arc;
use utils::shared_memory::handle::SharedMemoryHandle;

/// Process resource tracker for managing shared memory handles and cleanup
pub struct ProcessResourceTracker {
    /// Pod identifier for shared memory
    pub pod_identifier: String,
    /// Container name
    pub container_name: String,
    /// Shared memory handle (maintains reference count automatically)
    pub shared_memory_handle: Arc<SharedMemoryHandle>,
}

impl ProcessResourceTracker {
    pub fn new(
        pod_identifier: String,
        container_name: String,
        shared_memory_handle: Arc<SharedMemoryHandle>,
    ) -> Self {
        Self {
            pod_identifier,
            container_name,
            shared_memory_handle,
        }
    }
}
