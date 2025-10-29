use std::time::{SystemTime, UNIX_EPOCH};

use crate::ErlError;
use crate::backend::DeviceBackend;
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
    /// Cost scaling factor (applied after normalization to reduce cost)
    pub cost_scale: f64,
}

impl Default for KernelLimiterConfig {
    fn default() -> Self {
        Self {
            base_cost: 1.0,
            min_cost: 0.1,
            baseline_threads: 1024.0,
            // Scale down cost by 100x to make it more reasonable
            // A 4M-thread kernel will cost ~40 tokens instead of 4096
            cost_scale: 0.01,
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
    ///
    /// Note: This method only attempts to consume tokens. Token refilling
    /// is handled by the hypervisor side, which periodically adds tokens
    /// based on the refill_rate adjusted by the PID controller.
    pub fn try_acquire_at(
        &self,
        device: usize,
        grid_count: u32,
        block_count: u32,
        _now: f64,
    ) -> Result<bool, ErlError> {
        let cost = self.estimate_kernel_cost(grid_count, block_count);

        // Atomically consume tokens if available
        let tokens_before = self.backend.fetch_sub_tokens(device, cost)?;

        // Check if we had enough tokens before the subtraction
        Ok(tokens_before >= cost)
    }

    /// Inspect the current token count (without refilling).
    pub fn current_tokens(&self, device: usize) -> Result<f64, ErlError> {
        let state = self.backend.read_token_state(device)?;
        Ok(state.tokens)
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
        let raw_cost = self.cfg.base_cost * normalized * self.cfg.cost_scale;
        raw_cost.max(self.cfg.min_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DeviceBackend, DeviceQuota, TokenState};
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

        fn fetch_sub_tokens(&self, _device: usize, cost: f64) -> Result<f64, ErlError> {
            let mut state = self.state.lock().unwrap();
            let before = state.tokens;
            if before < cost {
                return Ok(before);
            }
            state.tokens = (state.tokens - cost).max(0.0);
            Ok(before)
        }

        fn fetch_add_tokens(&self, _device: usize, amount: f64) -> Result<f64, ErlError> {
            let mut state = self.state.lock().unwrap();
            let capacity = self.quota.lock().unwrap().capacity;
            let before = state.tokens;
            state.tokens = (state.tokens + amount).min(capacity).max(0.0);
            Ok(before)
        }
    }

    #[test]
    fn admits_when_tokens_available() {
        // With cost_scale=0.01, a kernel with 16×256=4096 threads costs:
        // (4096/1024) × 1.0 × 0.01 = 0.04 tokens
        // 5.0 tokens is more than enough
        let backend = MockBackend::new(10.0, 5.0, 5.0);
        let limiter = KernelLimiter::new(backend);
        let allowed = limiter.try_acquire_at(0, 16, 256, 1.0).unwrap();
        assert!(allowed, "should allow when tokens are sufficient");
    }

    #[test]
    fn denies_when_tokens_insufficient() {
        // With cost_scale=0.01, a kernel with 32×256=8192 threads costs:
        // (8192/1024) × 1.0 × 0.01 = 0.08 tokens
        // So we need very low tokens to trigger denial
        let backend = MockBackend::new(5.0, 0.0, 0.05);
        let limiter = KernelLimiter::new(backend);
        let allowed = limiter.try_acquire_at(0, 32, 256, 1.0).unwrap();
        assert!(!allowed, "should deny when tokens are insufficient");
    }

    #[test]
    fn tokens_remain_low_until_hypervisor_refills() {
        // Without hypervisor refilling, tokens won't automatically increase
        let backend = MockBackend::new(10.0, 2.0, 0.0);
        let limiter = KernelLimiter::new(backend);
        assert!(
            !limiter.try_acquire_at(0, 4, 64, 0.1).unwrap(),
            "should deny when tokens are zero"
        );
        // Time passing doesn't matter - hypervisor must refill
        assert!(
            !limiter.try_acquire_at(0, 4, 64, 10.0).unwrap(),
            "should still deny even after time passes"
        );
    }
}
