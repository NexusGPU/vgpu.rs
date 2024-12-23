use anyhow::Result;

mod worker;

/// GPU resource requirements
#[derive(Debug, Clone)]
pub struct GpuResources {
    /// GPU memory requirement (in bytes)
    pub memory_bytes: u64,
    /// GPU compute resource requirement (percentage 0-100)
    pub compute_percentage: u32,
}

/// Process state
#[derive(Debug, Clone, PartialEq)]
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
    /// Get process ID
    fn id(&self) -> u32;

    /// Get current process state
    fn state(&self) -> ProcessState;

    /// Get requested resources
    fn requested_resources(&self) -> GpuResources;

    /// Get current actual resource usage
    fn current_resources(&self) -> Result<GpuResources>;

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

        fn current_resources(&self) -> Result<GpuResources> {
            Ok(self.requested.clone())
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
