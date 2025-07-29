//! GPU Device State Watcher Module

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::time::Duration;

use error_stack::{Report, ResultExt};
use k8s_openapi::ClusterResourceScope;
use kube::Api;
use notify::{Config, Event, RecommendedWatcher, Watcher};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::{fs, select, time::sleep};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::infrastructure::k8s::types::KubernetesError;
use crate::infrastructure::kube_client;

/// GPU device state watcher
pub struct GpuDeviceStateWatcher {
    kubelet_device_state_path: PathBuf,
}

impl GpuDeviceStateWatcher {
    pub fn new(kubelet_device_state_path: PathBuf) -> Self {
        Self {
            kubelet_device_state_path,
        }
    }

    pub async fn run(
        &self,
        cancellation_token: CancellationToken,
        kubeconfig: Option<PathBuf>,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Starting GPU allocation watcher");
        Ok(())
    }
}
