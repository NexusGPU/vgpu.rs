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
#[derive(Debug, Clone, PartialEq)]
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

    /// Pause process (retain memory)
    fn pause(&self) -> Result<()>;

    /// Pause process and release memory
    fn release(&self) -> Result<()>;

    /// Resume process execution
    fn resume(&self) -> Result<()>;
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::RwLock;

    use super::*;

    // Mock implementation of GpuProcess for testing
    pub(crate) struct MockGpuProcess {
        id: u32,
        state: RwLock<ProcessState>,
        requested: GpuResources,
    }

    impl MockGpuProcess {
        pub(crate) fn new(id: u32, memory: u64, compute: u32) -> Self {
            Self {
                id,
                state: RwLock::new(ProcessState::Running),
                requested: GpuResources {
                    memory_bytes: memory,
                    compute_percentage: compute,
                },
            }
        }
    }

    impl GpuProcess for MockGpuProcess {
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
            self.requested.clone()
        }

        fn pause(&self) -> Result<()> {
            *self.state.write().expect("poisoned") = ProcessState::Paused;
            Ok(())
        }

        fn release(&self) -> Result<()> {
            *self.state.write().expect("poisoned") = ProcessState::Released;
            Ok(())
        }

        fn resume(&self) -> Result<()> {
            *self.state.write().expect("poisoned") = ProcessState::Running;
            Ok(())
        }

        fn gpu_uuid(&self) -> &str {
            "mock_uuid"
        }
    }

    #[test]
    fn test_gpu_resources() {
        let resources = GpuResources {
            memory_bytes: 1024,
            compute_percentage: 50,
        };
        assert_eq!(resources.memory_bytes, 1024);
        assert_eq!(resources.compute_percentage, 50);
    }

    #[test]
    fn test_process_state_equality() {
        assert_eq!(ProcessState::Running, ProcessState::Running);
        assert_ne!(ProcessState::Running, ProcessState::Paused);
        assert_ne!(ProcessState::Running, ProcessState::Released);
        assert_ne!(ProcessState::Paused, ProcessState::Released);
    }

    #[test]
    fn test_mock_gpu_process_basic() {
        let process = MockGpuProcess::new(1, 2048, 75);
        assert_eq!(process.id(), 1);
        assert_eq!(process.state(), ProcessState::Running);

        let resources = process.requested_resources();
        assert_eq!(resources.memory_bytes, 2048);
        assert_eq!(resources.compute_percentage, 75);
    }

    #[test]
    fn test_mock_gpu_process_state_transitions() -> Result<()> {
        let process = MockGpuProcess::new(2, 1024, 50);

        // Test pause
        process.pause()?;
        assert_eq!(process.state(), ProcessState::Paused);

        // Test release
        process.release()?;
        assert_eq!(process.state(), ProcessState::Released);

        // Test resume
        process.resume()?;
        assert_eq!(process.state(), ProcessState::Running);

        Ok(())
    }
}
