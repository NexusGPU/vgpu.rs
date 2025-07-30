use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use api_types::QosLevel;

pub mod worker;

// Public exports - remove self-references as types are already public

/// GPU resource requirements
#[derive(Debug, Clone, Default)]
pub struct GpuResources {
    /// GPU memory requirement (in bytes)
    pub(crate) memory_bytes: u64,
    /// GPU compute resource requirement (percentage 0-100)
    pub(crate) compute_percentage: u32,
    /// Requested TFLOPS for the workload (from Kubernetes annotations)
    #[allow(dead_code)]
    pub(crate) tflops_request: Option<f64>,
    /// Maximum TFLOPS limit for the workload (from Kubernetes annotations)
    #[allow(dead_code)]
    pub(crate) tflops_limit: Option<f64>,
    /// Maximum memory limit in bytes (from Kubernetes annotations)
    #[allow(dead_code)]
    pub(crate) memory_limit: Option<u64>,
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
pub trait GpuProcess: Send + Sync {
    /// Get process pid
    fn pid(&self) -> u32;

    /// Get process name
    fn name(&self) -> &str;

    /// Get current actual resource usage for each GPU
    fn current_resources(&self) -> HashMap<&str, GpuResources>;

    /// Get qos level
    fn qos_level(&self) -> QosLevel;

    /// Pause process (retain memory)
    fn pause(&self) -> Result<()>;

    /// Pause process and release memory
    fn release(&self) -> Result<()>;

    /// Resume process execution
    fn resume(&self) -> Result<()>;
}

impl<T: GpuProcess> GpuProcess for Arc<T> {
    fn pid(&self) -> u32 {
        self.as_ref().pid()
    }

    fn name(&self) -> &str {
        self.as_ref().name()
    }

    fn current_resources(&self) -> HashMap<&str, GpuResources> {
        self.as_ref().current_resources()
    }

    fn qos_level(&self) -> QosLevel {
        self.as_ref().qos_level()
    }

    fn pause(&self) -> Result<()> {
        self.as_ref().pause()
    }

    fn release(&self) -> Result<()> {
        self.as_ref().release()
    }

    fn resume(&self) -> Result<()> {
        self.as_ref().resume()
    }
}
