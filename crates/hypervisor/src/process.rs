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

    /// Get current process state
    fn state(&self) -> ProcessState;

    /// Get GPU UUID
    fn gpu_uuid(&self) -> &str;

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
