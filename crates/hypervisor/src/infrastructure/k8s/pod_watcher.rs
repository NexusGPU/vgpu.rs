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

use crate::infrastructure::k8s::pod_info::TensorFusionPodInfo;
use crate::infrastructure::k8s::types::KubernetesError;
use crate::infrastructure::k8s::types::WorkerUpdate;
use crate::infrastructure::kube_client;

/// Watches Kubernetes pods for tensor-fusion annotations.
///
/// This component monitors pod creation, updates, and deletion events,
/// extracting tensor-fusion specific annotations and sending updates
/// to the hypervisor system.
pub struct PodWatcher {
    namespace: Option<String>,
    node_name: String,
    kubeconfig: Option<PathBuf>,
}

impl PodWatcher {
    pub(crate) fn new(
        kubeconfig: Option<PathBuf>,
        namespace: Option<String>,
        node_name: String,
    ) -> Self {
        Self {
            namespace,
            node_name,
            kubeconfig,
        }
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
        update_sender: mpsc::Sender<WorkerUpdate>,
        cancellation_token: CancellationToken,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Starting pod watcher");
        let client = kube_client::init_kube_client(self.kubeconfig.clone()).await?;
        loop {
            select! {
                _ = cancellation_token.cancelled() => {
                    info!("Pod watcher shutdown requested");
                    break;
                }
                result = self.watch_pods(&client, &update_sender) => {
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

    /// Manually retrieve and parse pod information for a specific pod.
    ///
    /// This method fetches a pod by name and namespace, then extracts and validates
    /// tensor-fusion annotations to create a `TensorFusionPodInfo` object.
    ///
    /// # Arguments
    ///
    /// * `namespace` - The Kubernetes namespace of the pod
    /// * `pod_name` - The name of the pod to retrieve
    ///
    /// # Returns
    ///
    /// Returns `None` if the pod doesn't have tensor-fusion annotations,
    /// or `Some(TensorFusionPodInfo)` if annotations are found and valid.
    ///
    /// # Errors
    ///
    /// - [`KubernetesError::ConnectionFailed`] if unable to connect to Kubernetes API
    /// - [`KubernetesError::PodNotFound`] if the pod doesn't exist
    /// - [`KubernetesError::AnnotationParseError`] if annotation values are invalid
    pub async fn get_pod_info_manually(
        &self,
        namespace: &str,
        pod_name: &str,
    ) -> Result<Option<TensorFusionPodInfo>, Report<KubernetesError>> {
        // Initialize client for this operation
        let client = kube_client::init_kube_client(self.kubeconfig.clone()).await?;
        let api: Api<Pod> = Api::namespaced(client, namespace);

        let pod = api.get(pod_name).await.map_err(|e| {
            Report::new(KubernetesError::PodNotFound {
                pod_name: pod_name.to_string(),
                namespace: namespace.to_string(),
            })
            .attach_printable(format!("Kubernetes API error: {e}"))
        })?;

        // Transform the pod to TensorFusionPodInfo
        let mut tf_info = self.transform_pod_to_pod_info(pod).await?;

        // Only return the info if it has tensor-fusion annotations
        if tf_info.has_annotations() {
            // Ensure the basic pod information is set
            tf_info.0.namespace = namespace.to_string();
            tf_info.0.pod_name = pod_name.to_string();
            Ok(Some(tf_info))
        } else {
            Ok(None)
        }
    }

    /// Watch pods and process events.
    ///
    /// # Errors
    ///
    /// - [`KubernetesError::WatchFailed`] if the watch operation fails
    async fn watch_pods(
        &self,
        client: &Client,
        update_sender: &mpsc::Sender<WorkerUpdate>,
    ) -> Result<(), Report<KubernetesError>> {
        let api: Api<Pod> = match &self.namespace {
            Some(ns) => Api::namespaced(client.clone(), ns),
            None => Api::all(client.clone()),
        };

        let config = Config::default()
            .labels("tensor-fusion.ai/component=worker")
            .fields(&format!("spec.nodeName={}", self.node_name));

        let mut stream = watcher(api, config).applied_objects().boxed();

        while let Some(event) = stream.next().await {
            match event {
                Ok(pod) => {
                    if let Err(e) = self.handle_pod_event(pod, update_sender).await {
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

    async fn transform_pod_to_pod_info(
        &self,
        pod: Pod,
    ) -> Result<TensorFusionPodInfo, Report<KubernetesError>> {
        let metadata = pod.metadata;
        let pod_name = metadata.name.unwrap_or_else(|| "unknown".to_string());
        let namespace = metadata.namespace.unwrap_or_else(|| "default".to_string());
        let labels = metadata.labels.unwrap_or_default();

        // Extract annotations
        let annotations = metadata.annotations.unwrap_or_default();
        let mut tf_info = TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &labels)?;

        let node_name = pod.spec.and_then(|spec| spec.node_name);

        // Set pod metadata
        tf_info.0.node_name = node_name;
        tf_info.0.namespace = namespace;
        tf_info.0.pod_name = pod_name;

        Ok(tf_info)
    }
    /// Handle a single pod event.
    ///
    /// Extracts tensor-fusion annotations and sends appropriate updates
    /// if the pod contains relevant annotations.
    async fn handle_pod_event(
        &self,
        pod: Pod,
        update_sender: &mpsc::Sender<WorkerUpdate>,
    ) -> Result<(), Report<KubernetesError>> {
        // Clone the metadata to check deletion status while preserving pod ownership
        let is_being_deleted = pod.metadata.deletion_timestamp.is_some();
        let tf_info = self.transform_pod_to_pod_info(pod).await?;
        if is_being_deleted {
            let update = WorkerUpdate::PodDeleted {
                pod_name: tf_info.0.pod_name,
                namespace: tf_info.0.namespace,
            };

            if let Err(e) = update_sender.send(update).await {
                warn!("Failed to send worker update: {e}");
            }
        } else {
            if !tf_info.has_annotations() {
                return Ok(());
            }

            let update = WorkerUpdate::PodCreated { pod_info: tf_info };
            if let Err(e) = update_sender.send(update).await {
                warn!("Failed to send worker update: {e}");
            }
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
            namespace: None,
            node_name: "test-node".to_string(),
            kubeconfig: None,
        };

        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/tflops-request".to_string(),
            "10.0".to_string(),
        );

        let pod = create_test_pod("test-pod", annotations);

        watcher.handle_pod_event(pod, &tx).await.unwrap();

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
            namespace: None,
            node_name: "test-node".to_string(),
            kubeconfig: None,
        };

        let pod = create_test_pod("test-pod", BTreeMap::new());

        watcher.handle_pod_event(pod, &tx).await.unwrap();

        // Should not receive any update
        assert!(rx.try_recv().is_err());
    }
}
