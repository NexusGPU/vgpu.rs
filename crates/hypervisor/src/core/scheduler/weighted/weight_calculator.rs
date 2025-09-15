//! Weight calculation logic for the weighted scheduler

use super::types::WithTraps;
use crate::core::process::GpuProcess;
use api_types::QosLevel;

/// Trait for calculating process weights
pub trait Weight {
    fn weight(&self) -> u32;
}

/// Weight calculator for processes with traps
impl<T: GpuProcess> Weight for WithTraps<T> {
    fn weight(&self) -> u32 {
        let qos_multiplier = calculate_qos_multiplier(self.process.qos_level());
        let base_weight = qos_multiplier * 10;
        let trap_weight = self
            .traps
            .iter()
            .fold(0, |acc, trap| acc + (trap.round * qos_multiplier));

        base_weight + trap_weight
    }
}

/// Calculate QoS multiplier based on the level
fn calculate_qos_multiplier(qos: QosLevel) -> u32 {
    match qos {
        QosLevel::Low => 1,
        QosLevel::Medium => 2,
        QosLevel::High => 3,
        QosLevel::Critical => 4,
    }
}
