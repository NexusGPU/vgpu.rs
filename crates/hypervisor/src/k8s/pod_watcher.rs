use std::path::PathBuf;
use std::time::Duration;

use error_stack::Report;
use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::runtime::watcher::watcher;
use kube::runtime::watcher::Config;
use kube::runtime::WatchStreamExt;
use kube::Api;
use kube::Client;
use tokio::select;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::k8s::pod_info::TensorFusionPodInfo;
use crate::k8s::types::KubernetesError;
use crate::k8s::types::WorkerUpdate;
use crate::kube_client;

/// Watches Kubernetes pods for tensor-fusion annotations.
///
/// This component monitors pod creation, updates, and deletion events,
/// extracting tensor-fusion specific annotations and sending updates
/// to the hypervisor system.
pub(crate) struct PodWatcher {
    client: Client,
    namespace: Option<String>,
    node_name: String,
    update_sender: mpsc::Sender<WorkerUpdate>,
}

impl PodWatcher {
    /// Create a new pod watcher.
    ///
    /// # Arguments
    ///
    /// * `kubeconfig` - Optional path to kubeconfig file (None for default config)
    /// * `namespace` - Kubernetes namespace to watch (None for all namespaces)
    /// * `node_name` - Filter pods by node name
    /// * `update_sender` - Channel to send worker updates
    ///
    /// # Errors
    ///
    /// - [`KubernetesError::ConnectionFailed`] if unable to connect to Kubernetes API
    pub(crate) async fn new(
        kubeconfig: Option<PathBuf>,
        namespace: Option<String>,
        node_name: String,
        update_sender: mpsc::Sender<WorkerUpdate>,
    ) -> Result<Self, Report<KubernetesError>> {
        let client = kube_client::init_kube_client(kubeconfig).await?;

        Ok(Self {
            client,
            namespace,
            node_name,
            update_sender,
        })
    }

    /// Start watching pods for changes.
    ///
    /// This method runs indefinitely, watching for pod events and sending
    /// updates through the configured channel. It handles reconnection
    /// automatically if the watch stream fails.
    ///
    /// # Errors
    ///
    /// - [`KubernetesError::WatchFailed`] if the watch operation fails repeatedly
    #[tracing::instrument(skip(self, cancellation_token), fields(namespace = ?self.namespace, node_name = ?self.node_name))]
    pub(crate) async fn run(
        &self,
        cancellation_token: CancellationToken,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Starting pod watcher");

        loop {
            select! {
                _ = cancellation_token.cancelled() => {
                    info!("Pod watcher shutdown requested");
                    break;
                }
                result = self.watch_pods() => {
                    match result {
                        Ok(()) => {
                            warn!("Pod watch stream ended unexpectedly, restarting...");
                        }
                        Err(e) => {
                            error!("Pod watch failed: {e:?}");
                            // Wait before retrying
                            tokio::time::sleep(Duration::from_secs(5)).await;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Watch pods and process events.
    ///
    /// # Errors
    ///
    /// - [`KubernetesError::WatchFailed`] if the watch operation fails
    async fn watch_pods(&self) -> Result<(), Report<KubernetesError>> {
        let api: Api<Pod> = match &self.namespace {
            Some(ns) => Api::namespaced(self.client.clone(), ns),
            None => Api::all(self.client.clone()),
        };

        let config = Config::default()
            .labels("tensor-fusion.ai/component=worker")
            .fields(&format!("spec.nodeName={}", self.node_name));

        let mut stream = watcher(api, config).applied_objects().boxed();

        while let Some(event) = stream.next().await {
            match event {
                Ok(pod) => {
                    if let Err(e) = self.handle_pod_event(pod).await {
                        error!("Failed to handle pod event: {e:?}");
                    }
                }
                Err(e) => {
                    return Err(Report::new(KubernetesError::WatchFailed {
                        message: format!("Watch stream error: {e}"),
                    }));
                }
            }
        }

        Ok(())
    }

    /// Handle a single pod event.
    ///
    /// Extracts tensor-fusion annotations and sends appropriate updates
    /// if the pod contains relevant annotations.
    async fn handle_pod_event(&self, pod: Pod) -> Result<(), Report<KubernetesError>> {
        let metadata = pod.metadata;
        let pod_name = metadata.name.unwrap_or_else(|| "unknown".to_string());
        let namespace = metadata.namespace.unwrap_or_else(|| "default".to_string());
        let labels = metadata.labels.unwrap_or_default();

        // Extract annotations
        let annotations = metadata.annotations.unwrap_or_default();
        let mut tf_info = TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &labels)?;

        // Only process pods with tensor-fusion annotations
        if !tf_info.has_annotations() {
            return Ok(());
        }

        let node_name = pod.spec.and_then(|spec| spec.node_name);

        // Determine the type of event
        let update = if metadata.deletion_timestamp.is_some() {
            WorkerUpdate::PodDeleted {
                pod_name,
                namespace,
            }
        } else {
            tf_info.0.node_name = node_name;
            tf_info.0.namespace = namespace;
            tf_info.0.pod_name = pod_name;
            // For simplicity, treat all non-deleted pods as "created"
            // In a more sophisticated implementation, we could track pod generations
            // or resource versions to distinguish between creates and updates
            WorkerUpdate::PodCreated { pod_info: tf_info }
        };

        if let Err(e) = self.update_sender.send(update).await {
            warn!("Failed to send worker update: {e}");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use k8s_openapi::api::core::v1::Pod;
    use k8s_openapi::api::core::v1::PodSpec;
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use tokio::sync::mpsc;

    use super::*;

    fn create_test_pod(name: &str, annotations: BTreeMap<String, String>) -> Pod {
        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                namespace: Some("default".to_string()),
                annotations: Some(annotations),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        }
    }

    #[tokio::test]
    async fn handle_pod_event_with_annotations() {
        let (tx, mut rx) = mpsc::channel(3);
        let watcher = PodWatcher {
            client: Client::try_default().await.unwrap_or_else(|_| {
                // Skip test if no K8s cluster available
                Client::try_from(kube::Config::new("http://localhost:8080".parse().unwrap()))
                    .unwrap()
            }),
            namespace: None,
            node_name: "test-node".to_string(),
            update_sender: tx,
        };

        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/tflops-request".to_string(),
            "10.0".to_string(),
        );

        let pod = create_test_pod("test-pod", annotations);

        watcher.handle_pod_event(pod).await.unwrap();

        let update = rx.recv().await.unwrap();
        match update {
            WorkerUpdate::PodCreated { pod_info } => {
                assert_eq!(pod_info.0.pod_name, "test-pod");
            }
            _ => panic!("Expected PodCreated event"),
        }
    }

    #[tokio::test]
    async fn handle_pod_event_without_annotations() {
        let (tx, mut rx) = mpsc::channel(32);
        let watcher = PodWatcher {
            client: Client::try_default().await.unwrap_or_else(|_| {
                Client::try_from(kube::Config::new("http://localhost:8080".parse().unwrap()))
                    .unwrap()
            }),
            namespace: None,
            node_name: "test-node".to_string(),
            update_sender: tx,
        };

        let pod = create_test_pod("test-pod", BTreeMap::new());

        watcher.handle_pod_event(pod).await.unwrap();

        // Should not receive any update
        assert!(rx.try_recv().is_err());
    }
}
