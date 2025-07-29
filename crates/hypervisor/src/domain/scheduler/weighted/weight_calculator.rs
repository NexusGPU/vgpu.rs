//! Weight calculation logic for the weighted scheduler

use super::types::WithTraps;
use crate::process::GpuProcess;
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

/// Weight calculator utility functions
pub struct WeightCalculator;

impl WeightCalculator {
    /// Calculate effective priority for scheduling decisions
    pub fn effective_priority(qos: QosLevel, trap_rounds: u32) -> u32 {
        let base = calculate_qos_multiplier(qos);
        base * (10 + trap_rounds)
    }

    /// Check if process should be prioritized based on weight difference
    pub fn should_prioritize(current_weight: u32, candidate_weight: u32, threshold: u32) -> bool {
        candidate_weight.saturating_sub(current_weight) >= threshold
    }

    /// Calculate memory pressure impact on weight
    pub fn memory_pressure_adjustment(base_weight: u32, memory_pressure: f32) -> u32 {
        if memory_pressure > 0.8 {
            base_weight / 2 // Reduce weight under high memory pressure
        } else if memory_pressure > 0.6 {
            base_weight * 3 / 4
        } else {
            base_weight
        }
    }
}
