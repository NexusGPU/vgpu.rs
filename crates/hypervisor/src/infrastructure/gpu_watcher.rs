//! GPU Device State Watcher Module

use std::path::PathBuf;

use error_stack::Report;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::infrastructure::k8s::types::KubernetesError;

/// GPU device state watcher
pub struct GpuDeviceStateWatcher {
    _kubelet_device_state_path: PathBuf,
}

impl GpuDeviceStateWatcher {
    pub fn new(kubelet_device_state_path: PathBuf) -> Self {
        Self {
            _kubelet_device_state_path: kubelet_device_state_path,
        }
    }

    pub async fn run(
        &self,
        _cancellation_token: CancellationToken,
        _kubeconfig: Option<PathBuf>,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Starting GPU allocation watcher");
        Ok(())
    }
}
