//! Core type definitions for the hypervisor.

use std::sync::Arc;

use crate::hypervisor::Hypervisor;
use crate::pod_management::PodManager;
use crate::process::worker::TensorFusionWorker;
use crate::scheduler::weighted::WeightedScheduler;

/// Concrete worker type
pub type Worker = Arc<TensorFusionWorker>;

/// Concrete scheduler type
pub type Scheduler = WeightedScheduler<Worker>;

/// Concrete hypervisor type used throughout the application
pub type HypervisorType = Hypervisor<Worker, Scheduler>;

/// Concrete pod manager type
pub type PodManagerType = PodManager;
