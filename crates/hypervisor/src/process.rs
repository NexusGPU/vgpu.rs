use anyhow::Result;

pub(crate) mod worker;

/// GPU resource requirements
#[derive(Debug, Clone, Default)]
pub(crate) struct GpuResources {
    /// GPU memory requirement (in bytes)
    pub(crate) memory_bytes: u64,
    /// GPU compute resource requirement (percentage 0-100)
    pub(crate) compute_percentage: u32,
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

#[derive(Debug, Clone, PartialEq, Copy)]
pub(crate) enum QosLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Trait for GPU processes
pub(crate) trait GpuProcess: Send + Sync {
    /// Get process ID
    fn id(&self) -> u32;

    // /// Get current process state
    // fn state(&self) -> ProcessState;

    // /// Get GPU UUID
    // fn gpu_uuid(&self) -> &str;

    /// Get requested resources
    #[allow(unused)]
    fn requested_resources(&self) -> GpuResources;

    /// Get current actual resource usage
    fn current_resources(&self) -> GpuResources;

    /// Get qos level
    fn qos_level(&self) -> QosLevel;

    /// Pause process (retain memory)
    fn pause(&self) -> Result<()>;

    /// Pause process and release memory
    fn release(&self) -> Result<()>;

    /// Resume process execution
    fn resume(&self) -> Result<()>;
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
        fn id(&self) -> u32 {
            self.id
        }

        // fn state(&self) -> ProcessState {
        //     self.state
        // }

        // fn gpu_uuid(&self) -> &str {
        //     "mock-gpu-uuid"
        // }

        fn requested_resources(&self) -> GpuResources {
            GpuResources {
                memory_bytes: self.memory_bytes,
                compute_percentage: self.compute_percentage,
            }
        }

        fn current_resources(&self) -> GpuResources {
            GpuResources {
                memory_bytes: self.memory_bytes,
                compute_percentage: self.compute_percentage,
            }
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
