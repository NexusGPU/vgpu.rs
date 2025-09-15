//! Pod information cache with watch capabilities
//!
//! This component maintains an in-memory cache of pod information, populated by watching
//! Kubernetes pod events. It provides fast access to pod data and falls back to real-time
//! API queries when data is not available in cache.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use error_stack::Report;
use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::runtime::watcher::watcher;
use kube::runtime::watcher::Config;
use kube::runtime::WatchStreamExt;
use kube::Api;
use kube::Client;
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::platform::k8s::pod_info::TensorFusionPodInfo;
use crate::platform::k8s::KubernetesError;
use crate::platform::kube_client;

/// Key for identifying a pod in the cache
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct PodKey {
    namespace: String,
    name: String,
}

impl PodKey {
    fn new(namespace: String, name: String) -> Self {
        Self { namespace, name }
    }
}

/// Pod information cache that watches Kubernetes pods and maintains local state
///
/// This component serves as both a watcher and a cache, providing:
/// - Fast local access to pod information
/// - Automatic updates via watch events
/// - Fallback to real-time API queries for cache misses
/// - Event broadcasting to other components
pub struct PodInfoCache {
    namespace: Option<String>,
    node_name: String,
    /// Cached pod information: (namespace, pod_name) -> TensorFusionPodInfo
    cache: Arc<DashMap<PodKey, TensorFusionPodInfo>>,
    /// Kubernetes client for fallback queries
    client: Client,
}

impl PodInfoCache {
    pub async fn init(
        kubeconfig: Option<PathBuf>,
        namespace: Option<String>,
        node_name: String,
    ) -> Result<Self, Report<KubernetesError>> {
        Ok(Self {
            namespace,
            node_name,
            cache: Arc::new(DashMap::new()),
            client: kube_client::init_kube_client(kubeconfig).await?,
        })
    }

    /// Get pod information from cache or fetch from API server
    ///
    /// This method first checks the local cache. If the pod is not found,
    /// it queries the Kubernetes API server directly and updates the cache.
    pub async fn get_pod_info(
        &self,
        namespace: &str,
        pod_name: &str,
    ) -> Result<Option<TensorFusionPodInfo>, Report<KubernetesError>> {
        let key = PodKey::new(namespace.to_string(), pod_name.to_string());

        // Try cache first
        if let Some(pod_info) = self.cache.get(&key) {
            debug!(
                namespace = %namespace,
                pod_name = %pod_name,
                "Found pod info in cache"
            );
            return Ok(Some(pod_info.clone()));
        }

        // Cache miss - fetch from API server
        debug!(
            namespace = %namespace,
            pod_name = %pod_name,
            "Pod info not in cache, fetching from API server"
        );

        self.fetch_and_cache_pod_info(namespace, pod_name).await
    }

    /// Check if pod exists in cache
    pub fn contains_pod(&self, namespace: &str, pod_name: &str) -> bool {
        let key = PodKey::new(namespace.to_string(), pod_name.to_string());
        self.cache.contains_key(&key)
    }

    /// Remove pod from cache
    pub fn remove_pod(&self, namespace: &str, pod_name: &str) -> Option<TensorFusionPodInfo> {
        let key = PodKey::new(namespace.to_string(), pod_name.to_string());
        self.cache.remove(&key).map(|(_, v)| v)
    }

    /// Start watching pods and maintaining cache
    ///
    /// This method runs the watch loop and broadcasts events to subscribers.
    /// It also initializes the Kubernetes client for fallback queries.
    #[tracing::instrument(skip(self, cancellation_token), fields(namespace = ?self.namespace, node_name = ?self.node_name))]
    pub async fn run(
        &self,
        cancellation_token: CancellationToken,
    ) -> Result<(), Report<KubernetesError>> {
        info!("Starting pod info cache");

        loop {
            select! {
                _ = cancellation_token.cancelled() => {
                    info!("Pod info cache shutdown requested");
                    break;
                }
                result = self.watch_and_cache_pods() => {
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

    /// Fetch pod info from API server and update cache
    async fn fetch_and_cache_pod_info(
        &self,
        namespace: &str,
        pod_name: &str,
    ) -> Result<Option<TensorFusionPodInfo>, Report<KubernetesError>> {
        // Ensure client is initialized
        let client = &self.client;

        let api: Api<Pod> = Api::namespaced(client.clone(), namespace);

        let pod = api.get(pod_name).await.map_err(|e| {
            Report::new(KubernetesError::PodNotFound {
                pod_name: pod_name.to_string(),
                namespace: namespace.to_string(),
            })
            .attach_printable(format!("Kubernetes API error: {e}"))
        })?;

        // Transform and cache the pod info
        let mut tf_info = self.transform_pod_to_pod_info(pod)?;

        if tf_info.has_annotations() {
            // Ensure the basic pod information is set
            tf_info.0.namespace = namespace.to_string();
            tf_info.0.pod_name = pod_name.to_string();

            // Update cache
            let key = PodKey::new(namespace.to_string(), pod_name.to_string());
            self.cache.insert(key, tf_info.clone());

            info!(
                namespace = %namespace,
                pod_name = %pod_name,
                "Cached pod info from API server"
            );

            Ok(Some(tf_info))
        } else {
            Ok(None)
        }
    }

    /// Watch pods and maintain cache
    async fn watch_and_cache_pods(&self) -> Result<(), Report<KubernetesError>> {
        let client = &self.client;

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

    /// Handle a pod event and update cache
    async fn handle_pod_event(&self, pod: Pod) -> Result<(), Report<KubernetesError>> {
        let is_being_deleted = pod.metadata.deletion_timestamp.is_some();
        let tf_info = self.transform_pod_to_pod_info(pod)?;

        let key = PodKey::new(tf_info.0.namespace.clone(), tf_info.0.pod_name.clone());

        if is_being_deleted {
            // Remove from cache
            self.cache.remove(&key);
        } else {
            if !tf_info.has_annotations() {
                return Ok(());
            }
            // Update cache
            self.cache.insert(key, tf_info.clone());
        }
        Ok(())
    }

    fn transform_pod_to_pod_info(
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
}
