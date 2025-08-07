use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use api_types::QosLevel;

pub mod worker;

// Re-export worker type
pub use worker::TensorFusionWorker;

/// Concrete worker type
pub type Worker = Arc<TensorFusionWorker>;

// Public exports - remove self-references as types are already public

/// GPU resource requirements
#[derive(Debug, Clone, Default)]
pub struct GpuResources {
    /// GPU memory requirement (in bytes)
    pub(crate) memory_bytes: u64,
    /// GPU compute resource requirement (percentage 0-100)
    pub(crate) compute_percentage: u32,
}

/// Process state
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum ProcessState {
    /// Running
    Running,
    /// Paused (memory retained)
    Paused,
    /// Paused and memory released
    Released,
}

/// Trait for GPU processes
#[async_trait::async_trait]
pub trait GpuProcess: Send + Sync {
    /// Get process pid
    fn pid(&self) -> u32;

    /// Get process name
    fn name(&self) -> &str;

    /// Get current actual resource usage for each GPU
    async fn current_resources(&self) -> HashMap<&str, GpuResources>;

    /// Get qos level
    fn qos_level(&self) -> QosLevel;

    /// Pause process (retain memory)
    async fn pause(&self) -> Result<()>;

    /// Pause process and release memory
    async fn release(&self) -> Result<()>;

    /// Resume process execution
    async fn resume(&self) -> Result<()>;
}

#[async_trait::async_trait]
impl<T: GpuProcess> GpuProcess for Arc<T> {
    fn pid(&self) -> u32 {
        self.as_ref().pid()
    }

    fn name(&self) -> &str {
        self.as_ref().name()
    }

    async fn current_resources(&self) -> HashMap<&str, GpuResources> {
        self.as_ref().current_resources().await
    }

    fn qos_level(&self) -> QosLevel {
        self.as_ref().qos_level()
    }

    async fn pause(&self) -> Result<()> {
        self.as_ref().pause().await
    }

    async fn release(&self) -> Result<()> {
        self.as_ref().release().await
    }

    async fn resume(&self) -> Result<()> {
        self.as_ref().resume().await
    }
}
