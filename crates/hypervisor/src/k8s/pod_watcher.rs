use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use error_stack::Report;
use error_stack::ResultExt;
use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::config::KubeConfigOptions;
use kube::config::Kubeconfig;
use kube::runtime::watcher::watcher;
use kube::runtime::watcher::Config;
use kube::runtime::WatchStreamExt;
use kube::Api;
use kube::Client;
use tokio::select;
use tokio::sync::oneshot;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::k8s::annotations::TensorFusionAnnotations;
use crate::k8s::types::KubernetesError;
use crate::k8s::types::WorkerUpdate;

/// Watches Kubernetes pods for tensor-fusion annotations.
///
/// This component monitors pod creation, updates, and deletion events,
/// extracting tensor-fusion specific annotations and sending updates
/// to the hypervisor system.
pub(crate) struct PodWatcher {
    client: Client,
    namespace: Option<String>,
    node_name: Option<String>,
    update_sender: mpsc::Sender<WorkerUpdate>,
}

impl PodWatcher {
    /// Create a new pod watcher.
    ///
    /// # Arguments
    ///
    /// * `kubeconfig` - Optional path to kubeconfig file (None for default config)
    /// * `namespace` - Kubernetes namespace to watch (None for all namespaces)
    /// * `node_name` - Filter pods by node name (None for all nodes)
    /// * `update_sender` - Channel to send worker updates
    ///
    /// # Errors
    ///
    /// - [`KubernetesError::ConnectionFailed`] if unable to connect to Kubernetes API
    pub(crate) async fn new(
        kubeconfig: Option<PathBuf>,
        namespace: Option<String>,
        node_name: Option<String>,
        update_sender: mpsc::Sender<WorkerUpdate>,
    ) -> Result<Self, Report<KubernetesError>> {
        let client = match kubeconfig {
            Some(kubeconfig_path) => {
                // Load kubeconfig from the specified file
                let kubeconfig = Kubeconfig::read_from(&kubeconfig_path).change_context(
                    KubernetesError::ConnectionFailed {
                        message: format!(
                            "Failed to read kubeconfig file: {}",
                            kubeconfig_path.display()
                        ),
                    },
                )?;

                let config =
                    kube::Config::from_custom_kubeconfig(kubeconfig, &KubeConfigOptions::default())
                        .await
                        .change_context(KubernetesError::ConnectionFailed {
                            message: format!(
                                "Failed to create config from kubeconfig: {}",
                                kubeconfig_path.display()
                            ),
                        })?;

                Client::try_from(config).change_context(KubernetesError::ConnectionFailed {
                    message: "Failed to create Kubernetes client from custom kubeconfig"
                        .to_string(),
                })?
            }
            None => {
                // Use default configuration (in-cluster or ~/.kube/config)
                Client::try_default()
                    .await
                    .change_context(KubernetesError::ConnectionFailed {
                        message: "Failed to create Kubernetes client".to_string(),
                    })?
            }
        };

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
    #[tracing::instrument(skip(self, shutdown_rx), fields(namespace = ?self.namespace, node_name = ?self.node_name))]
    pub(crate) async fn run(
        &self,
        mut shutdown_rx: oneshot::Receiver<()>,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Starting pod watcher");

        loop {
            select! {
                _ = &mut shutdown_rx => {
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

        let mut config = Config::default();

        // Filter by node name if specified
        if let Some(node_name) = &self.node_name {
            config = config.fields(&format!("spec.nodeName={node_name}"));
        }

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

        // Extract annotations
        let annotations = metadata.annotations.unwrap_or_default();
        let tf_annotations = TensorFusionAnnotations::from_pod_annotations(&annotations)?;

        // Only process pods with tensor-fusion annotations
        if !tf_annotations.has_annotations() {
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
            // For simplicity, treat all non-deleted pods as "created"
            // In a more sophisticated implementation, we could track pod generations
            // or resource versions to distinguish between creates and updates
            WorkerUpdate::PodCreated {
                pod_name,
                namespace,
                annotations: tf_annotations,
                node_name,
            }
        };

        if let Err(e) = self.update_sender.send(update) {
            warn!("Failed to send worker update: {e}");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::mpsc;

    use k8s_openapi::api::core::v1::Pod;
    use k8s_openapi::api::core::v1::PodSpec;
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;

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
        let (tx, rx) = mpsc::channel();
        let watcher = PodWatcher {
            client: Client::try_default().await.unwrap_or_else(|_| {
                // Skip test if no K8s cluster available
                Client::try_from(kube::Config::new("http://localhost:8080".parse().unwrap()))
                    .unwrap()
            }),
            namespace: None,
            node_name: None,
            update_sender: tx,
        };

        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/tflops-request".to_string(),
            "10.0".to_string(),
        );

        let pod = create_test_pod("test-pod", annotations);

        watcher.handle_pod_event(pod).await.unwrap();

        let update = rx.recv().unwrap();
        match update {
            WorkerUpdate::PodCreated { pod_name, .. } => {
                assert_eq!(pod_name, "test-pod");
            }
            _ => panic!("Expected PodCreated event"),
        }
    }

    #[tokio::test]
    async fn handle_pod_event_without_annotations() {
        let (tx, rx) = mpsc::channel();
        let watcher = PodWatcher {
            client: Client::try_default().await.unwrap_or_else(|_| {
                Client::try_from(kube::Config::new("http://localhost:8080".parse().unwrap()))
                    .unwrap()
            }),
            namespace: None,
            node_name: None,
            update_sender: tx,
        };

        let pod = create_test_pod("test-pod", BTreeMap::new());

        watcher.handle_pod_event(pod).await.unwrap();

        // Should not receive any update
        assert!(rx.try_recv().is_err());
    }
}
