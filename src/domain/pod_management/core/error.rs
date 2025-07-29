//! Error types for pod management

use std::fmt;

/// Result type for pod management operations
pub type Result<T> = std::result::Result<T, PodManagementError>;

/// Comprehensive error type for pod management operations
#[derive(Debug)]
pub enum PodManagementError {
    /// Pod not found
    PodNotFound(String),
    /// Container not found
    ContainerNotFound(String),
    /// Worker not found
    WorkerNotFound(u32),
    /// Device allocation failed
    DeviceAllocationFailed(String),
    /// Resource limit exceeded
    ResourceLimitExceeded(String),
    /// Invalid configuration
    InvalidConfiguration(String),
    /// Shared memory operation failed
    SharedMemoryError(String),
    /// GPU operation failed
    GpuError(String),
    /// NVML operation failed
    NvmlError(nvml_wrapper::error::NvmlError),
    /// CUDA operation failed
    CudaError(String),
    /// IO operation failed
    IoError(std::io::Error),
    /// Other errors
    Other(String),
}

impl fmt::Display for PodManagementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PodNotFound(id) => write!(f, "Pod not found: {}", id),
            Self::ContainerNotFound(name) => write!(f, "Container not found: {}", name),
            Self::WorkerNotFound(pid) => write!(f, "Worker not found: {}", pid),
            Self::DeviceAllocationFailed(msg) => write!(f, "Device allocation failed: {}", msg),
            Self::ResourceLimitExceeded(msg) => write!(f, "Resource limit exceeded: {}", msg),
            Self::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::SharedMemoryError(msg) => write!(f, "Shared memory error: {}", msg),
            Self::GpuError(msg) => write!(f, "GPU error: {}", msg),
            Self::NvmlError(err) => write!(f, "NVML error: {}", err),
            Self::CudaError(msg) => write!(f, "CUDA error: {}", msg),
            Self::IoError(err) => write!(f, "IO error: {}", err),
            Self::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for PodManagementError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NvmlError(err) => Some(err),
            Self::IoError(err) => Some(err),
            _ => None,
        }
    }
}

// Convenient conversion implementations
impl From<nvml_wrapper::error::NvmlError> for PodManagementError {
    fn from(err: nvml_wrapper::error::NvmlError) -> Self {
        Self::NvmlError(err)
    }
}

impl From<std::io::Error> for PodManagementError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

impl From<anyhow::Error> for PodManagementError {
    fn from(err: anyhow::Error) -> Self {
        Self::Other(err.to_string())
    }
}

impl From<cudarc::driver::DriverError> for PodManagementError {
    fn from(err: cudarc::driver::DriverError) -> Self {
        Self::CudaError(err.to_string())
    }
}