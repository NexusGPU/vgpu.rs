use std::time::{SystemTime, UNIX_EPOCH};

use crate::ErlError;
use crate::backend::DeviceBackend;
use error_stack::Result;

/// Configuration for estimating kernel cost when updating the bucket.
#[derive(Debug, Clone)]
pub struct KernelLimiterConfig {
    /// Minimum number of tokens any kernel launch will consume.
    pub min_cost: f64,
    /// Maximum number of tokens a kernel launch can consume.
    pub max_cost: f64,
    /// Controls how quickly the sigmoid-style curve approaches `max_cost` as total threads grow.
    pub curve_scale: f64,
}

impl Default for KernelLimiterConfig {
    fn default() -> Self {
        Self {
            min_cost: 0.1,
            max_cost: 500.0,
            // Larger values make the curve flatter; 400k threads gets ~86% of max cost.
            curve_scale: 400_000.0,
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
        let scale = self.cfg.curve_scale.max(f64::EPSILON);
        let curve = 1.0 - (-(total_threads / scale)).exp();
        let span = (self.cfg.max_cost - self.cfg.min_cost).max(0.0);
        let cost = self.cfg.min_cost + curve * span;
        cost.max(self.cfg.min_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DeviceBackend, DeviceQuota, TokenState};
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone)]
    struct MockBackend {
        quota: Arc<Mutex<DeviceQuota>>,
        state: Arc<Mutex<TokenState>>,
    }

    impl MockBackend {
        fn new(capacity: f64, refill_rate: f64, tokens: f64) -> Self {
            Self {
                quota: Arc::new(Mutex::new(DeviceQuota::new(capacity, refill_rate))),
                state: Arc::new(Mutex::new(TokenState::new(tokens, 0.0))),
            }
        }

        fn tokens(&self) -> f64 {
            self.state.lock().unwrap().tokens
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
        // Curve-based model keeps cost in [min_cost, max_cost]; defaults give < 1 token here.
        // 5.0 tokens is more than enough
        let backend = MockBackend::new(10.0, 5.0, 5.0);
        let limiter = KernelLimiter::new(backend.clone());
        let allowed = limiter.try_acquire_at(0, 16, 256, 1.0).unwrap();
        assert!(allowed, "should allow when tokens are sufficient");
        assert!(
            backend.tokens() > 0.0,
            "should consume a bounded amount of tokens"
        );
    }

    #[test]
    fn denies_when_tokens_insufficient() {
        // Curve-based model still consumes at least min_cost per kernel.
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

    #[test]
    fn curve_cost_respects_bounds() {
        let cfg = KernelLimiterConfig {
            min_cost: 0.5,
            max_cost: 2.0,
            curve_scale: 1_000.0,
        };

        let backend = MockBackend::new(50.0, 5.0, 50.0);
        let limiter = KernelLimiter::with_config(backend.clone(), cfg.clone());
        let allowed = limiter.try_acquire_at(0, 1, 1, 0.0).unwrap();
        assert!(allowed, "should allow with ample tokens");

        let consumed_small = 50.0 - backend.tokens();
        assert!(
            consumed_small >= cfg.min_cost - 1e-6,
            "small kernels should respect min cost"
        );
        assert!(
            consumed_small <= cfg.max_cost + 1e-6,
            "small kernels should remain within bounds"
        );

        let backend_large = MockBackend::new(50.0, 5.0, 50.0);
        let limiter_large = KernelLimiter::with_config(backend_large.clone(), cfg);
        let allowed_large = limiter_large
            .try_acquire_at(0, 1_000_000, 2_048, 0.0)
            .unwrap();
        assert!(
            allowed_large,
            "should allow large kernels when tokens sufficient"
        );
        let consumed_large = 50.0 - backend_large.tokens();
        assert!(
            consumed_large <= 2.0 + 1e-6,
            "large kernels should not exceed max cost"
        );
        assert!(
            consumed_large >= 0.5 - 1e-6,
            "large kernels should still be >= min cost"
        );
    }
}
