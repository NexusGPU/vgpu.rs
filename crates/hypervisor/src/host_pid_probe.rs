//! Host PID probe module.
//!
//! This module provides functionality to discover host and container process ID pairs
//! for Kubernetes pods by scanning `/proc` filesystem. It implements a subscription-based
//! model where clients can subscribe to receive notifications when specific pod/container
//! combinations are found.
//!
//! # Key Components
//!
//! - [`HostPidProbe`]: Main service that manages subscriptions and scanning
//! - [`PodProcessInfo`]: Result type containing host and container PID pairs
//! - [`SubscriptionRequest`]: Configuration for what to monitor

use core::error::Error;
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use std::time::Duration;

use derive_more::Display;
use tokio::sync::oneshot;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

/// Information about a pod's process identifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PodProcessInfo {
    /// Host process ID
    pub host_pid: u32,
    /// Container process ID
    pub container_pid: u32,
    /// Pod name
    pub pod_name: String,
    /// Pod namespace
    pub namespace: String,
    /// Container name
    pub container_name: String,
}

/// Request to subscribe for a specific pod and container combination.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SubscriptionRequest {
    /// Name of the pod to monitor
    pub pod_name: String,
    /// Namespace of the pod
    pub namespace: String,
    /// Name of the container within the pod to monitor
    pub container_name: String,
    /// The PID within the container's namespace to find.
    pub container_pid: u32,
}

/// Errors that can occur during host PID probing operations.
#[derive(Debug, Display)]
pub enum HostPidProbeError {
    #[display("Failed to read proc filesystem: {message}")]
    ProcReadError { message: String },
    #[display("Failed to parse process information: {message}")]
    ParseError { message: String },
    #[display("Subscription channel was closed")]
    ChannelClosed,
}

impl Error for HostPidProbeError {}

type SubscriptionSender = oneshot::Sender<PodProcessInfo>;
type ActiveSubscriptions = Arc<Mutex<HashMap<SubscriptionRequest, SubscriptionSender>>>;

/// Host PID probe service that discovers process information for Kubernetes pods.
///
/// This service monitors the `/proc` filesystem to find processes belonging to
/// specific pod and container combinations. It uses a subscription model where
/// clients can request to be notified when a particular pod/container pair is
/// discovered.
///
/// # Performance
///
/// The service only runs periodic scans when there are active subscriptions.
/// Once all subscriptions are fulfilled or cancelled, the scanning stops automatically.
pub struct HostPidProbe {
    subscriptions: ActiveSubscriptions,
    scan_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
    scan_interval: Duration,
}

impl HostPidProbe {
    /// Creates a new host PID probe service.
    ///
    /// The `scan_interval` determines how frequently the service will scan
    /// the `/proc` filesystem when there are active subscriptions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::time::Duration;
    ///
    /// use hostpidprobe::HostPidProbe;
    ///
    /// let probe = HostPidProbe::new(Duration::from_secs(1));
    /// ```
    pub fn new(scan_interval: Duration) -> Self {
        Self {
            subscriptions: Arc::new(Mutex::new(HashMap::new())),
            scan_handle: Arc::new(Mutex::new(None)),
            scan_interval,
        }
    }

    /// Subscribes to receive notification for a single, specific process.
    ///
    /// The `request` must contain a `container_pid`. Returns a [`oneshot::Receiver`]
    /// that will receive a [`PodProcessInfo`] when the requested pod, container,
    /// and container PID combination is discovered.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use hostpidprobe::{HostPidProbe, SubscriptionRequest};
    /// # use std::time::Duration;
    /// # async fn example() {
    /// let probe = HostPidProbe::new(Duration::from_secs(1));
    /// let request = SubscriptionRequest {
    ///     pod_name: "my-pod".to_string(),
    ///     namespace: "my-namespace".to_string(),
    ///     container_name: "my-container".to_string(),
    ///     container_pid: 42,
    /// };
    ///
    /// let receiver = probe.subscribe(request).await;
    /// match receiver.await {
    ///     Ok(info) => println!("Found process: {:?}", info),
    ///     Err(_) => println!("Subscription was cancelled"),
    /// }
    /// # }
    /// ```
    #[tracing::instrument(level = "debug", skip(self))]
    #[allow(clippy::async_yields_async)]
    pub async fn subscribe(
        &self,
        request: SubscriptionRequest,
    ) -> oneshot::Receiver<PodProcessInfo> {
        let (sender, receiver) = oneshot::channel();

        let mut guard = self.subscriptions.lock().await;
        let is_first = guard.is_empty();
        guard.insert(request.clone(), sender);
        drop(guard);

        info!(
            pod_name = %request.pod_name,
            container_name = %request.container_name,
            container_pid = %request.container_pid,
            "Added subscription"
        );

        if is_first {
            self.start_scanning().await;
        }

        receiver
    }

    /// Starts the periodic scanning process.
    ///
    /// This method is called automatically when the first subscription is added.
    #[tracing::instrument(level = "debug", skip(self))]
    async fn start_scanning(&self) {
        let mut scan_handle = self.scan_handle.lock().await;

        if scan_handle.is_some() {
            return;
        }

        info!("Starting periodic process scanning");

        let subscriptions = Arc::clone(&self.subscriptions);
        let scan_handle_ref = Arc::clone(&self.scan_handle);
        let interval = self.scan_interval;

        let handle = tokio::spawn(async move {
            let mut interval_timer = time::interval(interval);

            loop {
                interval_timer.tick().await;

                let should_continue = Self::scan_and_notify(Arc::clone(&subscriptions)).await;

                if !should_continue {
                    debug!("No active subscriptions, stopping scan");
                    break;
                }
            }

            // Clear the scan handle when done
            let mut handle_guard = scan_handle_ref.lock().await;
            *handle_guard = None;
        });

        *scan_handle = Some(handle);
    }

    /// Performs a single scan of the proc filesystem and notifies subscribers.
    ///
    /// Returns `true` if scanning should continue (there are active subscriptions),
    /// `false` if scanning should stop.
    #[tracing::instrument(level = "trace", skip(subscriptions))]
    async fn scan_and_notify(subscriptions: ActiveSubscriptions) -> bool {
        let found_processes = match Self::scan_proc_filesystem().await {
            Ok(processes) => processes,
            Err(e) => {
                error!(error = %e, "Failed to scan proc filesystem");
                return true; // Continue scanning despite error
            }
        };

        let mut subscriptions_guard = subscriptions.lock().await;
        if subscriptions_guard.is_empty() {
            return false;
        }

        if found_processes.is_empty() {
            return true; // Keep scanning if there are subscriptions
        }

        let mut fulfilled_requests = Vec::new();

        for process in found_processes {
            let key = SubscriptionRequest {
                pod_name: process.pod_name.clone(),
                namespace: process.namespace.clone(),
                container_name: process.container_name.clone(),
                container_pid: process.container_pid,
            };

            if subscriptions_guard.contains_key(&key) {
                fulfilled_requests.push((key, process));
            }
        }

        for (key, process_info) in fulfilled_requests {
            if let Some(sender) = subscriptions_guard.remove(&key) {
                if sender.send(process_info).is_err() {
                    warn!(?key, "Receiver dropped for subscription");
                } else {
                    info!(?key, "Notified subscriber");
                }
            }
        }

        // Return true if there are still active subscriptions
        !subscriptions_guard.is_empty()
    }

    /// Scans the proc filesystem to find all pod processes.
    ///
    /// # Errors
    ///
    /// - [`HostPidProbeError::ProcReadError`] if the proc filesystem cannot be read
    /// - [`HostPidProbeError::ParseError`] if process information cannot be parsed
    #[tracing::instrument(level = "trace")]
    async fn scan_proc_filesystem() -> Result<Vec<PodProcessInfo>, HostPidProbeError> {
        let mut processes = Vec::new();

        let proc_dir = match fs::read_dir("/proc") {
            Ok(dir) => dir,
            Err(e) => {
                return Err(HostPidProbeError::ProcReadError {
                    message: format!("Cannot read /proc directory: {e}"),
                });
            }
        };

        for entry in proc_dir {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => continue,
            };

            let file_name = entry.file_name();
            let pid_str = match file_name.to_str() {
                Some(name) => name,
                None => continue,
            };

            // Check if this is a PID directory
            let pid: u32 = match pid_str.parse() {
                Ok(pid) => pid,
                Err(_) => continue,
            };

            if let Ok(process_info) = Self::extract_process_info(pid).await {
                processes.push(process_info);
            }
        }

        debug!(
            found_processes = processes.len(),
            "Completed proc filesystem scan"
        );
        Ok(processes)
    }

    /// Extracts process information for a specific PID.
    ///
    /// # Errors
    ///
    /// - [`HostPidProbeError::ProcReadError`] if process files cannot be read
    /// - [`HostPidProbeError::ParseError`] if process information cannot be parsed
    #[tracing::instrument(level = "trace")]
    async fn extract_process_info(pid: u32) -> Result<PodProcessInfo, HostPidProbeError> {
        let environ_path = format!("/proc/{pid}/environ");
        let status_path = format!("/proc/{pid}/status");

        // Read environment variables
        let environ_data =
            fs::read_to_string(&environ_path).map_err(|e| HostPidProbeError::ProcReadError {
                message: format!("Cannot read {environ_path}: {e}"),
            })?;

        // Extract pod name, namespace, and container name from environment
        let (pod_name, namespace, container_name) =
            Self::parse_environment_variables(&environ_data)?;

        // Read status file to get namespace PID
        let status_data =
            fs::read_to_string(&status_path).map_err(|e| HostPidProbeError::ProcReadError {
                message: format!("Cannot read {status_path}: {e}"),
            })?;

        let container_pid = Self::parse_container_pid(&status_data)?;

        Ok(PodProcessInfo {
            host_pid: pid,
            container_pid,
            pod_name,
            namespace,
            container_name,
        })
    }

    /// Parses environment variables to extract pod name, namespace, and container name.
    ///
    /// # Errors
    ///
    /// - [`HostPidProbeError::ParseError`] if required environment variables are not found
    fn parse_environment_variables(
        environ_data: &str,
    ) -> Result<(String, String, String), HostPidProbeError> {
        let mut pod_name = None;
        let mut namespace = None;
        let mut container_name = None;

        for env_var in environ_data.split('\0') {
            if let Some(value) = env_var.strip_prefix("POD_NAME=") {
                pod_name = Some(value.to_string());
            } else if let Some(value) = env_var.strip_prefix("POD_NAMESPACE=") {
                namespace = Some(value.to_string());
            } else if let Some(value) = env_var.strip_prefix("CONTAINER_NAME=") {
                container_name = Some(value.to_string());
            }
        }

        match (pod_name, namespace, container_name) {
            (Some(pod), Some(ns), Some(container)) => Ok((pod, ns, container)),
            _ => Err(HostPidProbeError::ParseError {
                message: "POD_NAME, POD_NAMESPACE, or CONTAINER_NAME not found in environment"
                    .to_string(),
            }),
        }
    }

    /// Parses the status file to extract the container PID from NSpid.
    ///
    /// # Errors
    ///
    /// - [`HostPidProbeError::ParseError`] if NSpid line cannot be found or parsed
    fn parse_container_pid(status_data: &str) -> Result<u32, HostPidProbeError> {
        for line in status_data.lines() {
            if let Some(nspid_part) = line.strip_prefix("NSpid:") {
                let pids: Vec<&str> = nspid_part.split_whitespace().collect();

                // NSpid contains PIDs in different namespaces
                // The last PID is typically the one in the container namespace
                if let Some(&last_pid_str) = pids.last() {
                    return last_pid_str
                        .parse()
                        .map_err(|e| HostPidProbeError::ParseError {
                            message: format!("Cannot parse container PID from NSpid: {e}"),
                        });
                }
            }
        }

        Err(HostPidProbeError::ParseError {
            message: "NSpid not found in status file".to_string(),
        })
    }

    /// Stops the scanning process and clears all subscriptions.
    ///
    /// This method is useful for graceful shutdown.
    pub async fn shutdown(&self) {
        info!("Shutting down host PID probe");

        // Clear all subscriptions
        {
            let mut subscriptions = self.subscriptions.lock().await;
            subscriptions.clear();
        }

        // Stop the scanning task
        let mut scan_handle = self.scan_handle.lock().await;
        if let Some(handle) = scan_handle.take() {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn new_probe_has_no_active_subscriptions() {
        let probe = HostPidProbe::new(Duration::from_millis(100));
        let subscriptions = probe.subscriptions.lock().await;
        assert!(
            subscriptions.is_empty(),
            "New probe should have no subscriptions"
        );
    }

    #[tokio::test]
    async fn subscription_adds_to_active_list() {
        let probe = HostPidProbe::new(Duration::from_millis(100));
        let request = SubscriptionRequest {
            pod_name: "test-pod".to_string(),
            namespace: "test-namespace".to_string(),
            container_name: "test-container".to_string(),
            container_pid: 123,
        };

        let _receiver = probe.subscribe(request.clone()).await;

        let subscriptions = probe.subscriptions.lock().await;
        assert!(
            subscriptions.contains_key(&request),
            "Subscription should be added"
        );
    }

    #[test]
    fn parse_environment_variables_success() {
        let environ_data =
            "PATH=/usr/bin\0POD_NAME=my-pod\0POD_NAMESPACE=my-namespace\0CONTAINER_NAME=my-container\0HOME=/root\0";

        let result = HostPidProbe::parse_environment_variables(environ_data);

        assert!(
            result.is_ok(),
            "Should parse environment variables successfully"
        );
        let (pod_name, namespace, container_name) = result.unwrap();
        assert_eq!(pod_name, "my-pod");
        assert_eq!(namespace, "my-namespace");
        assert_eq!(container_name, "my-container");
    }

    #[test]
    fn parse_environment_variables_missing_pod_name() {
        let environ_data = "PATH=/usr/bin\0CONTAINER_NAME=my-container\0HOME=/root\0";

        let result = HostPidProbe::parse_environment_variables(environ_data);

        assert!(result.is_err(), "Should fail when POD_NAME is missing");
    }

    #[test]
    fn parse_environment_variables_missing_container_name() {
        let environ_data =
            "PATH=/usr/bin\0POD_NAME=my-pod\0POD_NAMESPACE=my-namespace\0HOME=/root\0";

        let result = HostPidProbe::parse_environment_variables(environ_data);

        assert!(
            result.is_err(),
            "Should fail when CONTAINER_NAME is missing"
        );
    }

    #[test]
    fn parse_environment_variables_missing_namespace() {
        let environ_data =
            "PATH=/usr/bin\0POD_NAME=my-pod\0CONTAINER_NAME=my-container\0HOME=/root\0";

        let result = HostPidProbe::parse_environment_variables(environ_data);

        assert!(result.is_err(), "Should fail when POD_NAMESPACE is missing");
    }

    #[test]
    fn subscription_request_with_different_namespaces() {
        let request1 = SubscriptionRequest {
            pod_name: "pod1".to_string(),
            namespace: "namespace1".to_string(),
            container_name: "container1".to_string(),
            container_pid: 1,
        };

        let request2 = SubscriptionRequest {
            pod_name: "pod1".to_string(),
            namespace: "namespace2".to_string(),
            container_name: "container1".to_string(),
            container_pid: 1,
        };

        assert_ne!(
            request1, request2,
            "Requests with different namespaces should not be equal"
        );
    }

    #[test]
    fn pod_process_info_with_different_namespaces() {
        let info1 = PodProcessInfo {
            host_pid: 1234,
            container_pid: 1,
            pod_name: "pod1".to_string(),
            namespace: "namespace1".to_string(),
            container_name: "container1".to_string(),
        };

        let info2 = PodProcessInfo {
            host_pid: 1234,
            container_pid: 1,
            pod_name: "pod1".to_string(),
            namespace: "namespace2".to_string(),
            container_name: "container1".to_string(),
        };

        assert_ne!(
            info1, info2,
            "Process info with different namespaces should not be equal"
        );
    }

    #[test]
    fn parse_environment_variables_with_extra_vars() {
        let environ_data = "PATH=/usr/bin\0POD_NAME=my-pod\0POD_NAMESPACE=my-namespace\0CONTAINER_NAME=my-container\0HOME=/root\0EXTRA_VAR=extra_value\0";

        let result = HostPidProbe::parse_environment_variables(environ_data);

        assert!(
            result.is_ok(),
            "Should parse environment variables successfully even with extra vars"
        );
        let (pod_name, namespace, container_name) = result.unwrap();
        assert_eq!(pod_name, "my-pod");
        assert_eq!(namespace, "my-namespace");
        assert_eq!(container_name, "my-container");
    }

    #[test]
    fn parse_container_pid_success() {
        let status_data = "Name:\ttesting\nPid:\t1234\nNSpid:\t1234\t1\nOther:\tvalue\n";

        let result = HostPidProbe::parse_container_pid(status_data);

        assert!(result.is_ok(), "Should parse container PID successfully");
        assert_eq!(result.unwrap(), 1, "Should extract the last PID from NSpid");
    }

    #[test]
    fn parse_container_pid_multiple_namespaces() {
        let status_data = "Name:\ttesting\nPid:\t1234\nNSpid:\t1234\t567\t1\nOther:\tvalue\n";

        let result = HostPidProbe::parse_container_pid(status_data);

        assert!(result.is_ok(), "Should parse container PID successfully");
        assert_eq!(result.unwrap(), 1, "Should extract the last PID from NSpid");
    }

    #[test]
    fn parse_container_pid_missing_nspid() {
        let status_data = "Name:\ttesting\nPid:\t1234\nOther:\tvalue\n";

        let result = HostPidProbe::parse_container_pid(status_data);

        assert!(result.is_err(), "Should fail when NSpid is missing");
    }

    #[test]
    fn parse_container_pid_invalid_format() {
        let status_data = "Name:\ttesting\nPid:\t1234\nNSpid:\tinvalid\nOther:\tvalue\n";

        let result = HostPidProbe::parse_container_pid(status_data);

        assert!(result.is_err(), "Should fail when NSpid format is invalid");
    }

    #[test]
    fn subscription_request_equality() {
        let request1 = SubscriptionRequest {
            pod_name: "pod1".to_string(),
            namespace: "namespace1".to_string(),
            container_name: "container1".to_string(),
            container_pid: 1,
        };

        let request2 = SubscriptionRequest {
            pod_name: "pod1".to_string(),
            namespace: "namespace1".to_string(),
            container_name: "container1".to_string(),
            container_pid: 1,
        };

        let request3 = SubscriptionRequest {
            pod_name: "pod2".to_string(),
            namespace: "namespace1".to_string(),
            container_name: "container1".to_string(),
            container_pid: 1,
        };

        assert_eq!(request1, request2, "Identical requests should be equal");
        assert_ne!(request1, request3, "Different requests should not be equal");
    }

    #[test]
    fn pod_process_info_equality() {
        let info1 = PodProcessInfo {
            host_pid: 1234,
            container_pid: 1,
            pod_name: "pod1".to_string(),
            namespace: "namespace1".to_string(),
            container_name: "container1".to_string(),
        };

        let info2 = PodProcessInfo {
            host_pid: 1234,
            container_pid: 1,
            pod_name: "pod1".to_string(),
            namespace: "namespace1".to_string(),
            container_name: "container1".to_string(),
        };

        assert_eq!(info1, info2, "Identical process info should be equal");
    }

    #[tokio::test]
    async fn shutdown_clears_subscriptions() {
        let probe = HostPidProbe::new(Duration::from_millis(100));
        let request = SubscriptionRequest {
            pod_name: "test-pod".to_string(),
            namespace: "test-namespace".to_string(),
            container_name: "test-container".to_string(),
            container_pid: 123,
        };

        let _receiver = probe.subscribe(request).await;

        // Verify subscription was added
        {
            let subscriptions = probe.subscriptions.lock().await;
            assert!(!subscriptions.is_empty(), "Should have active subscription");
        }

        probe.shutdown().await;

        // Verify subscriptions were cleared
        {
            let subscriptions = probe.subscriptions.lock().await;
            assert!(
                subscriptions.is_empty(),
                "Subscriptions should be cleared after shutdown"
            );
        }
    }
}
