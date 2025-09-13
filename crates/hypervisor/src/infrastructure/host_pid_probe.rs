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
use std::sync::Arc;
use std::time::Duration;

use derive_more::Display;
use tokio::fs;
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
    /// # use hypervisor::host_pid_probe::HostPidProbe;
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
    /// The `request` must contain a `container_pid`.
    ///
    /// * `ttl` – Maximum duration to wait for a matching process. If the TTL
    ///   expires before a match is found, the receiver side will get an
    ///   `Err(RecvError)` indicating timeout/cancellation.
    ///
    /// Returns a [`oneshot::Receiver`] that will yield a [`PodProcessInfo`] once
    /// the requested pod/container PID pair is discovered, or error after TTL.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use hypervisor::host_pid_probe::{HostPidProbe, SubscriptionRequest};
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
    /// let receiver = probe.subscribe(request, Duration::from_secs(5)).await;
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
        ttl: Duration,
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

        // run ttl timer to remove subscription
        let subs_clone = Arc::clone(&self.subscriptions);
        let req_clone = request.clone();
        tokio::spawn(async move {
            time::sleep(ttl).await;
            let mut guard = subs_clone.lock().await;
            // if still exists, remove and automatically drop sender (Receiver will receive Err)
            guard.remove(&req_clone);
        });

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
        let found_processes = if cfg!(test) {
            Vec::new() // Mock: No processes found during tests
        } else {
            match Self::scan_proc_filesystem().await {
                Ok(processes) => processes,
                Err(e) => {
                    error!(error = %e, "Failed to scan proc filesystem");
                    return true; // Continue scanning despite error
                }
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

        let mut proc_dir = match fs::read_dir("/proc").await {
            Ok(dir) => dir,
            Err(e) => {
                return Err(HostPidProbeError::ProcReadError {
                    message: format!("Cannot read /proc directory: {e}"),
                });
            }
        };

        loop {
            let entry_opt = match proc_dir.next_entry().await {
                Ok(opt) => opt,
                Err(e) => {
                    return Err(HostPidProbeError::ProcReadError {
                        message: format!("Cannot read /proc entry: {e}"),
                    });
                }
            };

            // Break when no more entries
            let entry = match entry_opt {
                Some(ent) => ent,
                None => break,
            };

            let file_name = entry.file_name();
            let pid_str = match file_name.to_str() {
                Some(pid_str) => pid_str,
                None => continue,
            };

            let pid = match pid_str.parse::<u32>() {
                Ok(pid) => pid,
                Err(_) => continue,
            };

            match HostPidProbe::extract_process_info(pid).await {
                Ok(process_info) => processes.push(process_info),
                Err(e) => warn!(pid = %pid, error = %e, "failed to extract process info"),
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
        let environ_data = fs::read_to_string(&environ_path).await.map_err(|e| {
            HostPidProbeError::ProcReadError {
                message: format!("Cannot read {environ_path}: {e}"),
            }
        })?;

        // Extract pod name, namespace, and container name from environment
        let (pod_name, namespace, container_name) =
            Self::parse_environment_variables(&environ_data)?;

        // Read status file to get namespace PID
        let status_data = fs::read_to_string(&status_path).await.map_err(|e| {
            HostPidProbeError::ProcReadError {
                message: format!("Cannot read {status_path}: {e}"),
            }
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

    use tokio::time::timeout;

    use super::*;

    // set a unified timeout for async waiting to avoid test hanging
    const TEST_TIMEOUT: Duration = Duration::from_secs(1);

    #[test]
    fn parse_environment_variables_success() {
        let environ_data = "PATH=/usr/bin\0POD_NAME=my-pod\0POD_NAMESPACE=my-namespace\0CONTAINER_NAME=my-container\0HOME=/root\0";

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
    fn parse_environment_variables_missing_required() {
        // missing POD_NAME
        let environ_data = "PATH=/usr/bin\0CONTAINER_NAME=my-container\0HOME=/root\0";
        assert!(HostPidProbe::parse_environment_variables(environ_data).is_err());

        // missing CONTAINER_NAME
        let environ_data = "POD_NAME=my-pod\0POD_NAMESPACE=my-namespace\0";
        assert!(HostPidProbe::parse_environment_variables(environ_data).is_err());

        // missing POD_NAMESPACE
        let environ_data = "POD_NAME=my-pod\0CONTAINER_NAME=my-container\0";
        assert!(HostPidProbe::parse_environment_variables(environ_data).is_err());
    }

    #[test]
    fn parse_container_pid_variations() {
        // normal case
        let status_data = "Name:\ttesting\nPid:\t1234\nNSpid:\t1234\t1\n";
        assert_eq!(HostPidProbe::parse_container_pid(status_data).unwrap(), 1);

        // multiple namespaces
        let status_data = "NSpid:\t1234\t567\t10\n";
        assert_eq!(HostPidProbe::parse_container_pid(status_data).unwrap(), 10);

        // missing NSpid
        let status_data = "Name:\ttesting\n";
        assert!(HostPidProbe::parse_container_pid(status_data).is_err());
    }

    // Helper to generate subscription request & corresponding process info
    fn make_req(pid: u32) -> SubscriptionRequest {
        SubscriptionRequest {
            pod_name: "test-pod-nonexistent-12345".to_string(),
            namespace: "test-namespace-nonexistent-12345".to_string(),
            container_name: "test-container-nonexistent-12345".to_string(),
            container_pid: pid,
        }
    }

    #[tokio::test]
    async fn subscribe_notifies_when_manual_process_is_sent() {
        let probe = HostPidProbe::new(Duration::from_secs(30)); // set scan interval to avoid real scanning

        let request = make_req(42);
        let receiver = probe
            .subscribe(request.clone(), Duration::from_millis(100))
            .await;

        // manually send matching process info to sender, simulating scanning to target process
        {
            let mut subs = probe.subscriptions.lock().await;
            let sender = subs.remove(&request).expect("Subscription must exist");

            let info = PodProcessInfo {
                host_pid: 1000,
                container_pid: 42,
                pod_name: request.pod_name.clone(),
                namespace: request.namespace.clone(),
                container_name: request.container_name.clone(),
            };

            sender.send(info).expect("Failed to send process info");
        }

        let recv_info = timeout(TEST_TIMEOUT, receiver)
            .await
            .expect("Receiver timed out")
            .expect("Channel closed unexpectedly");

        assert_eq!(recv_info.container_pid, 42);
        assert_eq!(recv_info.namespace, "test-namespace-nonexistent-12345");
    }

    #[tokio::test]
    async fn subscribe_times_out_when_no_process_found() {
        let probe = HostPidProbe::new(Duration::from_millis(10));

        let ttl = Duration::from_millis(50);
        // Use a very unique PID that's extremely unlikely to exist
        let receiver = probe.subscribe(make_req(999_999_999), ttl).await;

        // no process info sent, expect timeout or TTL expiration
        let res = timeout(ttl + ttl, receiver).await;
        match res {
            Ok(Err(_)) => {
                // TTL expired, sender was dropped - this is expected
            }
            Err(_) => {
                // Timeout occurred - also acceptable
            }
            Ok(Ok(_)) => {
                panic!("Receiver should not get a successful message without matching process");
            }
        }
    }

    #[tokio::test]
    async fn multiple_subscriptions_all_get_notified() {
        let probe = HostPidProbe::new(Duration::from_secs(30));

        let req1 = make_req(1);
        let req2 = make_req(2);

        let recv1 = probe
            .subscribe(req1.clone(), Duration::from_millis(100))
            .await;
        let recv2 = probe
            .subscribe(req2.clone(), Duration::from_millis(100))
            .await;

        {
            let mut subs = probe.subscriptions.lock().await;

            // simulate scanning to first process
            if let Some(sender) = subs.remove(&req1) {
                sender
                    .send(PodProcessInfo {
                        host_pid: 100,
                        container_pid: 1,
                        pod_name: req1.pod_name.clone(),
                        namespace: req1.namespace.clone(),
                        container_name: req1.container_name.clone(),
                    })
                    .unwrap();
            }

            // simulate scanning to second process
            if let Some(sender) = subs.remove(&req2) {
                sender
                    .send(PodProcessInfo {
                        host_pid: 101,
                        container_pid: 2,
                        pod_name: req2.pod_name.clone(),
                        namespace: req2.namespace.clone(),
                        container_name: req2.container_name.clone(),
                    })
                    .unwrap();
            }
        }

        let info1 = timeout(TEST_TIMEOUT, recv1).await.unwrap().unwrap();
        let info2 = timeout(TEST_TIMEOUT, recv2).await.unwrap().unwrap();

        assert_eq!(info1.container_pid, 1);
        assert_eq!(info2.container_pid, 2);
    }

    #[tokio::test]
    async fn shutdown_clears_subscriptions() {
        let probe = HostPidProbe::new(Duration::from_millis(50));
        let _receiver = probe
            .subscribe(make_req(123), Duration::from_millis(100))
            .await;

        // ensure subscription exists
        {
            let subs = probe.subscriptions.lock().await;
            assert!(!subs.is_empty());
        }

        probe.shutdown().await;

        // after shutdown, subscriptions should be cleared
        {
            let subs = probe.subscriptions.lock().await;
            assert!(subs.is_empty());
        }
    }
}
