use std::time::{SystemTime, UNIX_EPOCH};

use crate::backend::{DeviceBackend, TokenState};
use crate::{DeviceQuota, ErlError};
use error_stack::Result;

/// Configuration for estimating kernel cost when updating the bucket.
#[derive(Debug, Clone)]
pub struct KernelLimiterConfig {
    /// Base cost in tokens when workload equals `baseline_threads`.
    pub base_cost: f64,
    /// Minimum number of tokens any kernel launch will consume.
    pub min_cost: f64,
    /// Normalization factor for `grid_count * block_count`.
    pub baseline_threads: f64,
}

impl Default for KernelLimiterConfig {
    fn default() -> Self {
        Self {
            base_cost: 1.0,
            min_cost: 0.25,
            baseline_threads: 1024.0,
        }
    }
}

/// Token-bucket limiter that can be called from the CUDA launch hook.
#[derive(Debug)]
pub struct KernelLimiter<B: DeviceBackend> {
    backend: B,
    cfg: KernelLimiterConfig,
}

impl<B: DeviceBackend> KernelLimiter<B> {
    pub fn new(backend: B) -> Self {
        Self::with_config(backend, KernelLimiterConfig::default())
    }

    pub fn with_config(backend: B, cfg: KernelLimiterConfig) -> Self {
        Self { backend, cfg }
    }

    /// Attempt to consume tokens at the current time.
    pub fn try_acquire(
        &self,
        device: usize,
        grid_count: u32,
        block_count: u32,
    ) -> Result<bool, ErlError> {
        let now = Self::now_seconds();
        self.try_acquire_at(device, grid_count, block_count, now)
    }

    /// Deterministic variant useful for testing.
    pub fn try_acquire_at(
        &self,
        device: usize,
        grid_count: u32,
        block_count: u32,
        now: f64,
    ) -> Result<bool, ErlError> {
        let quota = self.backend.read_quota(device)?;
        let mut state = self.backend.read_token_state(device)?;
        let refill = refill_tokens(&quota, &state, now);
        state.tokens = refill;
        state.last_update = now;

        let cost = self.estimate_kernel_cost(grid_count, block_count);

        let decision = if refill >= cost {
            state.tokens = (refill - cost).max(0.0);
            true
        } else {
            false
        };

        self.backend
            .write_token_state(device, TokenState::new(state.tokens, now))?;
        Ok(decision)
    }

    /// Inspect the bucket after applying refill.
    pub fn current_tokens(&self, device: usize) -> Result<f64, ErlError> {
        let quota = self.backend.read_quota(device)?;
        let state = self.backend.read_token_state(device)?;
        Ok(refill_tokens(&quota, &state, Self::now_seconds()))
    }

    fn now_seconds() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    fn estimate_kernel_cost(&self, grid_count: u32, block_count: u32) -> f64 {
        let total_threads = (grid_count.max(1) as f64) * (block_count.max(1) as f64);
        let normalized = if self.cfg.baseline_threads <= f64::EPSILON {
            total_threads
        } else {
            total_threads / self.cfg.baseline_threads
        };
        (self.cfg.base_cost * normalized).max(self.cfg.min_cost)
    }
}

fn refill_tokens(quota: &DeviceQuota, state: &TokenState, now: f64) -> f64 {
    let dt = (now - state.last_update).max(0.0);
    let replenished = state.tokens.max(0.0) + quota.refill_rate.max(0.0) * dt;
    replenished.min(quota.capacity.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DeviceBackend;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct MockBackend {
        quota: Mutex<DeviceQuota>,
        state: Mutex<TokenState>,
    }

    impl MockBackend {
        fn new(capacity: f64, refill_rate: f64, tokens: f64) -> Self {
            Self {
                quota: Mutex::new(DeviceQuota::new(capacity, refill_rate)),
                state: Mutex::new(TokenState::new(tokens, 0.0)),
            }
        }
    }

    impl DeviceBackend for MockBackend {
        fn read_token_state(&self, _device: usize) -> Result<TokenState, ErlError> {
            Ok(*self.state.lock().unwrap())
        }

        fn write_token_state(&self, _device: usize, state: TokenState) -> Result<(), ErlError> {
            *self.state.lock().unwrap() = state;
            Ok(())
        }

        fn read_quota(&self, _device: usize) -> Result<DeviceQuota, ErlError> {
            Ok(*self.quota.lock().unwrap())
        }

        fn write_refill_rate(&self, _device: usize, refill_rate: f64) -> Result<(), ErlError> {
            self.quota.lock().unwrap().refill_rate = refill_rate;
            Ok(())
        }

        fn write_capacity(&self, _device: usize, capacity: f64) -> Result<(), ErlError> {
            self.quota.lock().unwrap().capacity = capacity;
            Ok(())
        }
    }

    #[test]
    fn admits_when_tokens_available() {
        let backend = MockBackend::new(10.0, 5.0, 5.0);
        let limiter = KernelLimiter::new(backend);
        let allowed = limiter.try_acquire_at(0, 16, 256, 1.0).unwrap();
        assert!(allowed);
    }

    #[test]
    fn denies_when_tokens_insufficient() {
        let backend = MockBackend::new(5.0, 0.0, 0.2);
        let limiter = KernelLimiter::new(backend);
        let allowed = limiter.try_acquire_at(0, 32, 256, 1.0).unwrap();
        assert!(!allowed);
    }

    #[test]
    fn refills_over_time() {
        let backend = MockBackend::new(10.0, 2.0, 0.0);
        let limiter = KernelLimiter::new(backend);
        assert!(!limiter.try_acquire_at(0, 4, 64, 0.1).unwrap());
        assert!(limiter.try_acquire_at(0, 4, 64, 0.6).unwrap());
    }
}
