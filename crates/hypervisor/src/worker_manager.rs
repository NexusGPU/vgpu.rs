//! Worker management module.
#![allow(dead_code)]
//! This module provides functionality for managing workers based on Kubernetes pod events.
//! It replaces the old socket-based worker watcher with a pod-centric approach where
//! pods are treated as workers.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use api_types::QosLevel;
use api_types::WorkerInfo;
use tokio::sync::RwLock;
use tracing::info;
use tracing::warn;

use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::host_pid_probe::SubscriptionRequest;
use crate::k8s::TensorFusionAnnotations;
use crate::process::worker::TensorFusionWorker;

/// Entry that combines Kubernetes annotation info and optional running worker instance
#[derive(Clone)]
pub struct WorkerEntry {
    pub info: WorkerInfo,
    pub worker: Option<Arc<TensorFusionWorker>>,
}

impl WorkerEntry {
    fn new(info: WorkerInfo) -> Self {
        Self { info, worker: None }
    }
}

/// Worker registry for storing and managing worker information.
pub type WorkerRegistry = Arc<RwLock<HashMap<String, WorkerEntry>>>;

/// Worker manager that handles worker lifecycle based on pod events.
pub struct WorkerManager<AddCB, RemoveCB> {
    registry: WorkerRegistry,
    add_callback: AddCB,
    remove_callback: RemoveCB,
    host_pid_probe: Arc<HostPidProbe>,
}

impl<AddCB, RemoveCB> WorkerManager<AddCB, RemoveCB>
where
    AddCB: Fn(u32, Arc<TensorFusionWorker>) + Send + Sync + 'static,
    RemoveCB: Fn(u32) + Send + Sync + 'static,
{
    /// Create a new worker manager with host PID probe.
    pub fn new(
        host_pid_probe: Arc<HostPidProbe>,
        add_callback: AddCB,
        remove_callback: RemoveCB,
    ) -> Self {
        Self {
            registry: Arc::new(RwLock::new(HashMap::new())),
            add_callback,
            remove_callback,
            host_pid_probe,
        }
    }

    /// Get the worker registry for API queries.
    pub fn registry(&self) -> &WorkerRegistry {
        &self.registry
    }

    /// Handle a pod creation event.
    pub async fn handle_pod_created(
        &self,
        pod_name: String,
        namespace: String,
        annotations: TensorFusionAnnotations,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<()> {
        let worker_key = format!("{namespace}/{pod_name}");
        info!("Processing pod creation: {worker_key}");

        // Store worker info in registry
        {
            let mut registry = self.registry.write().await;
            registry.insert(worker_key.clone(), WorkerEntry::new(annotations.0.clone()));
            info!("Added worker to registry: {worker_key}");
        }

        // Start PID discovery for the pod
        self.discover_worker_pid(pod_name, namespace, gpu_observer)
            .await?;

        Ok(())
    }

    /// Discover worker PID using HostPidProbe and automatically associate it.
    async fn discover_worker_pid(
        &self,
        pod_name: String,
        namespace: String,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<()> {
        let worker_key = format!("{namespace}/{pod_name}");

        // Collect all container names for this pod. If the list is empty or not present,
        // fall back to using the pod name as a single container.
        let container_names = {
            let registry = self.registry.read().await;
            if let Some(entry) = registry.get(&worker_key) {
                // Clone the container list if present; otherwise use an empty vec.
                let mut names = entry.info.containers.clone().unwrap_or_default();
                if names.is_empty() {
                    names.push(pod_name.clone());
                }
                names
            } else {
                warn!("Worker not found in registry: {worker_key}");
                return Err(anyhow::anyhow!("Worker not found in registry"));
            }
        };

        for container_name in container_names {
            let subscription_request = SubscriptionRequest {
                pod_name: pod_name.clone(),
                container_name: container_name.clone(),
            };

            info!(
                pod_name = %pod_name,
                namespace = %namespace,
                container_name = %container_name,
                "Starting PID discovery for worker"
            );

            let receiver = self.host_pid_probe.subscribe(subscription_request).await;

            // Handle PID discovery result without spawning a task (sequential processing)
            let process_info = receiver
                .await
                .map_err(|_| anyhow::anyhow!("PID discovery subscription was cancelled"))?;

            info!(
                pod_name = %pod_name,
                namespace = %namespace,
                container_name = %container_name,
                host_pid = process_info.host_pid,
                container_pid = process_info.container_pid,
                "Discovered worker PID"
            );

            // Associate the worker with the discovered PID
            self.associate_discovered_worker(
                pod_name.clone(),
                namespace.clone(),
                process_info.host_pid,
                gpu_observer.clone(),
            )
            .await?;
        }

        Ok(())
    }

    /// Associate a worker with a discovered PID.
    async fn associate_discovered_worker(
        &self,
        pod_name: String,
        namespace: String,
        host_pid: u32,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<()> {
        let worker_key = format!("{namespace}/{pod_name}");

        let mut registry = self.registry.write().await;
        if let Some(entry) = registry.get_mut(&worker_key) {
            let WorkerInfo {
                namespace,
                pod_name,
                node_name: _,
                gpu_uuids,
                qos_level,
                tflops_request: _,
                tflops_limit: _,
                vram_request: _,
                vram_limit: _,
                containers: _,
            } = &entry.info;

            let gpu_uuids_vec = gpu_uuids.clone().unwrap_or_default();
            let qos = qos_level.unwrap_or(QosLevel::Medium);

            let worker = Arc::new(TensorFusionWorker::new(
                host_pid,
                qos,
                gpu_uuids_vec,
                gpu_observer,
                namespace.clone(),
                pod_name.clone(),
            ));

            entry.worker = Some(worker.clone());
            (self.add_callback)(host_pid, worker);

            info!("Associated worker {worker_key} with PID {host_pid}");
        } else {
            warn!("Attempted to associate PID with non-existent worker: {worker_key}");
            return Err(anyhow::anyhow!("Worker not found in registry"));
        }

        Ok(())
    }

    /// Handle a pod update event.
    pub async fn handle_pod_updated(
        &self,
        pod_name: String,
        namespace: String,
        annotations: TensorFusionAnnotations,
        node_name: Option<String>,
    ) -> Result<()> {
        let worker_key = format!("{namespace}/{pod_name}");
        info!("Processing pod update: {worker_key}");

        // For now, treat update the same as creation
        // In the future, we might want to handle updates differently
        {
            let mut registry = self.registry.write().await;
            if let Some(entry) = registry.get_mut(&worker_key) {
                entry.info.tflops_request = annotations.0.tflops_request;
                entry.info.tflops_limit = annotations.0.tflops_limit;
                entry.info.vram_request = annotations.0.vram_request;
                entry.info.vram_limit = annotations.0.vram_limit;
                entry.info.node_name = node_name;
                info!("Updated worker in registry: {worker_key}");
            } else {
                warn!("Attempted to update non-existent worker: {worker_key}");
            }
        }

        Ok(())
    }

    /// Handle a pod deletion event.
    pub async fn handle_pod_deleted(&self, pod_name: String, namespace: String) -> Result<()> {
        let worker_key = format!("{namespace}/{pod_name}");
        info!("Processing pod deletion: {worker_key}");

        {
            let mut registry = self.registry.write().await;
            if let Some(entry) = registry.remove(&worker_key) {
                info!("Removed worker from registry: {worker_key}");

                // Call remove callback if worker has an associated PID
                if let Some(worker) = &entry.worker {
                    use crate::process::GpuProcess;
                    (self.remove_callback)(worker.pid());
                }
            } else {
                warn!("Attempted to remove non-existent worker: {worker_key}");
            }
        }

        Ok(())
    }
}
