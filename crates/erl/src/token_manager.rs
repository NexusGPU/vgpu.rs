//! Token manager implementation
//!
//! Execute workload-aware token consumption in CUDA application process

use error_stack::Result;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::traits::{ErlError, SharedStorage, TokenManager};
use crate::workload_calc::{PowerWorkloadCalculator, WorkloadCalculator};

/// Workload-aware token manager
///
/// Run in limiter process, responsible for:
/// 1. Read avg_cost from shared memory
/// 2. Calculate dynamic cost based on workload
/// 3. Execute token bucket admission control
#[derive(Debug)]
pub struct WorkloadTokenManager<K, S>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
    S: SharedStorage<K>,
{
    /// Shared storage
    storage: S,
    /// Workload calculator
    workload_calculator: Box<dyn WorkloadCalculator>,
    /// Type marker
    _phantom: std::marker::PhantomData<K>,
}

impl<K, S> WorkloadTokenManager<K, S>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
    S: SharedStorage<K>,
{
    /// Create new token manager
    pub fn new(storage: S, workload_calculator: Box<dyn WorkloadCalculator>) -> Self {
        Self {
            storage,
            workload_calculator,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create new token manager with default PowerWorkloadCalculator
    pub fn with_default_calculator(storage: S) -> Self {
        Self::new(storage, Box::new(PowerWorkloadCalculator::default()))
    }

    /// Execute token bucket refill logic
    fn refill_tokens(&self, key: &K) -> Result<f64, ErlError> {
        let (capacity, refill_rate) = self.storage.load_quota(key)?;
        let (mut tokens, last_timestamp) = self.storage.load_token_state(key)?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let delta_time = now - last_timestamp;
        let tokens_to_add = refill_rate * delta_time;
        tokens = (tokens + tokens_to_add).min(capacity);

        self.storage.save_token_state(key, tokens, now)?;
        Ok(tokens)
    }
}

impl<K, S> TokenManager<K> for WorkloadTokenManager<K, S>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug,
    S: SharedStorage<K>,
{
    type Storage = S;

    fn try_acquire_workload(
        &mut self,
        key: &K,
        grid_count: u32,
        block_count: u32,
    ) -> Result<(), ErlError> {
        // 1. Execute refill
        let current_tokens = self.refill_tokens(key)?;

        // 2. Read current avg_cost from shared memory (updated by hypervisor)
        let base_avg_cost = self.storage.load_avg_cost(key).unwrap_or(1.0);

        // 3. Calculate dynamic cost based on workload
        let workload_factor = self
            .workload_calculator
            .calculate_factor(grid_count, block_count);
        let dynamic_cost = base_avg_cost * workload_factor;

        // 4. Check and deduct tokens
        if current_tokens >= dynamic_cost {
            let new_tokens = current_tokens - dynamic_cost;
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();

            self.storage.save_token_state(key, new_tokens, now)?;

            tracing::debug!(
                key = ?key,
                grid_count = grid_count,
                block_count = block_count,
                workload_factor = workload_factor,
                base_avg_cost = base_avg_cost,
                dynamic_cost = dynamic_cost,
                remaining_tokens = new_tokens,
                "Workload acquisition successful"
            );

            Ok(())
        } else {
            tracing::warn!(
                key = ?key,
                grid_count = grid_count,
                block_count = block_count,
                dynamic_cost = dynamic_cost,
                available_tokens = current_tokens,
                "Workload acquisition failed: insufficient tokens"
            );

            Err(error_stack::report!(ErlError::AdmissionDenied {
                reason: format!(
                    "Insufficient tokens for {:?}: need {:.3}, have {:.3} (grid: {}, block: {})",
                    key, dynamic_cost, current_tokens, grid_count, block_count
                )
            }))
        }
    }

    fn get_token_status(&self, key: &K) -> Result<(f64, f64, f64), ErlError> {
        let (capacity, refill_rate) = self.storage.load_quota(key)?;
        let current_tokens = self.refill_tokens(key)?;
        Ok((current_tokens, capacity, refill_rate))
    }
}
