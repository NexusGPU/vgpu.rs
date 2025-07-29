//! Core type definitions for the hypervisor.

use crate::hypervisor::Hypervisor;
use crate::pod_management::PodManager;
use crate::process::worker::TensorFusionWorker;
use crate::scheduler::weighted::WeightedScheduler;

/// Concrete hypervisor type used throughout the application
pub type HypervisorType = Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>;

/// Concrete pod manager type
pub type PodManagerType = PodManager;
