use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use api_types::QosLevel;

pub(crate) mod worker;

/// GPU resource requirements
#[derive(Debug, Clone, Default)]
pub(crate) struct GpuResources {
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
pub(crate) enum ProcessState {
    /// Running
    Running,
    /// Paused (memory retained)
    Paused,
    /// Paused and memory released
    Released,
}

/// Trait for GPU processes
pub(crate) trait GpuProcess: Send + Sync {
    /// Get process pid
    fn pid(&self) -> u32;

    /// Get process name
    fn name(&self) -> String;

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

    fn name(&self) -> String {
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) struct MockGpuProcess {
        id: u32,
        memory_bytes: u64,
        compute_percentage: u32,
        // state: ProcessState,
        qos_level: QosLevel,
    }

    impl MockGpuProcess {
        pub(crate) fn new(id: u32, memory_bytes: u64, compute_percentage: u32) -> Self {
            Self {
                id,
                memory_bytes,
                compute_percentage,
                // state: ProcessState::Running,
                qos_level: QosLevel::Medium,
            }
        }

        pub(crate) fn new_with_qos(
            id: u32,
            memory_bytes: u64,
            compute_percentage: u32,
            qos_level: QosLevel,
        ) -> Self {
            Self {
                id,
                memory_bytes,
                compute_percentage,
                // state: ProcessState::Running,
                qos_level,
            }
        }
    }

    impl GpuProcess for MockGpuProcess {
        fn pid(&self) -> u32 {
            self.id
        }

        fn name(&self) -> String {
            "mock-gpu-process".to_string()
        }

        fn current_resources(&self) -> HashMap<&str, GpuResources> {
            HashMap::new()
        }

        fn qos_level(&self) -> QosLevel {
            self.qos_level
        }

        fn pause(&self) -> Result<()> {
            Ok(())
        }

        fn release(&self) -> Result<()> {
            Ok(())
        }

        fn resume(&self) -> Result<()> {
            Ok(())
        }
    }

    unsafe impl Send for MockGpuProcess {}
    unsafe impl Sync for MockGpuProcess {}
}
