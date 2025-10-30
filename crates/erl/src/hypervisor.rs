use crate::ErlError;
use crate::backend::DeviceBackend;
use error_stack::Result;

/// Configuration for the PID-based device controller.
#[derive(Debug, Clone)]
pub struct DeviceControllerConfig {
    /// Target GPU utilization (0.0 to 1.0, e.g., 0.5 = 50%)
    pub target_utilization: f64,
    /// Minimum refill rate (tokens/second) - prevents rate from dropping to zero
    pub rate_min: f64,
    /// Maximum refill rate (tokens/second)
    pub rate_max: f64,

    /// PID proportional gain - how aggressively to respond to error
    pub kp: f64,
    /// PID integral gain - how quickly to eliminate steady-state error
    pub ki: f64,
    /// PID derivative gain - how much to dampen oscillations
    pub kd: f64,

    /// Low-pass filter coefficient for smoothing utilization (0.0 to 1.0)
    /// Higher values = less filtering (more responsive, more noise)
    pub filter_alpha: f64,

    /// Burst window in seconds - capacity = refill_rate × burst_window
    pub burst_window: f64,
    /// Minimum capacity (tokens)
    pub capacity_min: f64,
    /// Maximum capacity (tokens) - prevents unbounded growth
    pub capacity_max: f64,

    /// Minimum time between updates (seconds)
    pub min_delta_time: f64,
}

impl Default for DeviceControllerConfig {
    fn default() -> Self {
        Self {
            target_utilization: 0.5,
            rate_min: 10.0,
            rate_max: 100_000.0,

            // Gentle PID parameters for stability
            kp: 0.5,
            ki: 0.1,
            kd: 0.05,

            // Moderate filtering to smooth NVML noise
            filter_alpha: 0.3,

            // 2-second burst window
            burst_window: 2.0,
            capacity_min: 100.0,
            capacity_max: 200_000.0,

            min_delta_time: 0.05,
        }
    }
}

impl DeviceControllerConfig {
    /// Create a new config with specified target utilization and rate limits.
    pub fn new(target_utilization: f64, rate_max: f64) -> Self {
        Self {
            target_utilization,
            rate_max,
            capacity_max: rate_max * 2.0,
            ..Default::default()
        }
    }
}

/// Snapshot of controller state after an update.
#[derive(Debug, Clone)]
pub struct DeviceControllerState {
    pub target_utilization: f64,
    pub smoothed_utilization: f64,
    pub current_rate: f64,
    pub current_capacity: f64,
    pub token_drain_rate: f64,
}

/// PID-based controller that dynamically adjusts token refill rates.
///
/// # Algorithm Overview
///
/// 1. **Low-pass filter**: Smooth incoming utilization measurements using EMA
/// 2. **Drain rate estimation**: Calculate token consumption rate from bucket level changes
/// 3. **Base rate calculation**: Estimate ideal refill rate based on target utilization
/// 4. **PID correction**: Apply proportional-integral-derivative control to fine-tune
/// 5. **Capacity adjustment**: Scale capacity with refill rate (bounded)
#[derive(Debug)]
pub struct DeviceController<B: DeviceBackend> {
    backend: B,
    device: usize,
    cfg: DeviceControllerConfig,

    // PID state
    integral: f64,
    last_error: f64,

    // Filtering state
    smoothed_util: Option<f64>,

    // Rate tracking
    current_rate: f64,

    // Drain rate estimation
    last_token_level: f64,
    last_timestamp: Option<f64>,
}

impl<B: DeviceBackend> DeviceController<B> {
    pub fn new(backend: B, device: usize, cfg: DeviceControllerConfig) -> Result<Self, ErlError> {
        // Validate configuration
        if !(0.0..=1.0).contains(&cfg.target_utilization) {
            return Err(error_stack::report!(ErlError::invalid_config(
                "target_utilization must be in [0, 1]"
            )));
        }
        if cfg.rate_min <= 0.0 || cfg.rate_max <= cfg.rate_min {
            return Err(error_stack::report!(ErlError::invalid_config(
                "rate_max must be greater than rate_min > 0"
            )));
        }
        if !(0.0..=1.0).contains(&cfg.filter_alpha) {
            return Err(error_stack::report!(ErlError::invalid_config(
                "filter_alpha must be in [0, 1]"
            )));
        }

        // Initialize with a conservative starting rate
        let start_rate = 100.0_f64.min(cfg.rate_max).max(cfg.rate_min);
        let initial_capacity =
            (start_rate * cfg.burst_window).clamp(cfg.capacity_min, cfg.capacity_max);

        // Initialize backend
        backend.write_capacity(device, initial_capacity)?;
        backend.write_refill_rate(device, start_rate)?;

        let mut token_state = backend.read_token_state(device)?;
        token_state.tokens = initial_capacity;
        backend.write_token_state(device, token_state)?;

        tracing::debug!(
            device = device,
            target_util = %format!("{:.1}%", cfg.target_utilization * 100.0),
            initial_rate = %format!("{:.1}", start_rate),
            initial_capacity = %format!("{:.1}", initial_capacity),
            "Initialized ERL controller"
        );

        Ok(Self {
            backend,
            device,
            cfg,
            integral: 0.0,
            last_error: 0.0,
            smoothed_util: None,
            current_rate: start_rate,
            last_token_level: initial_capacity,
            last_timestamp: None,
        })
    }

    pub fn state(&self) -> DeviceControllerState {
        DeviceControllerState {
            target_utilization: self.cfg.target_utilization,
            smoothed_utilization: self.smoothed_util.unwrap_or(0.0),
            current_rate: self.current_rate,
            current_capacity: (self.current_rate * self.cfg.burst_window)
                .clamp(self.cfg.capacity_min, self.cfg.capacity_max),
            token_drain_rate: 0.0, // Will be updated during next cycle
        }
    }

    /// Update controller with new utilization measurement.
    fn update_internal(
        &mut self,
        measured_util: f64,
        delta_time: f64,
    ) -> Result<DeviceControllerState, ErlError> {
        let measured = measured_util.clamp(0.0, 1.0);

        // Step 1: Low-pass filter to smooth NVML noise
        let smoothed = self.smooth_utilization(measured);

        // Step 2: Estimate token drain rate
        let drain_rate = self.estimate_drain_rate(delta_time)?;

        // Step 3: Calculate base rate from drain rate and target
        let base_rate = self.calculate_base_rate(smoothed, drain_rate);

        // Step 4: Compute PID correction
        let error = self.cfg.target_utilization - smoothed;
        let correction = self.compute_pid_correction(error, delta_time);

        // Step 5: Apply correction to base rate
        let new_rate = (base_rate * (1.0 + correction)).clamp(self.cfg.rate_min, self.cfg.rate_max);
        self.current_rate = new_rate;

        // Step 6: Calculate capacity (bounded)
        let new_capacity =
            (new_rate * self.cfg.burst_window).clamp(self.cfg.capacity_min, self.cfg.capacity_max);

        // Step 7: Refill tokens
        let refill_amount = new_rate * delta_time;
        self.backend.fetch_add_tokens(self.device, refill_amount)?;

        // Step 8: Update backend
        self.backend.write_refill_rate(self.device, new_rate)?;
        self.backend.write_capacity(self.device, new_capacity)?;

        // Step 9: Clamp tokens to capacity
        self.clamp_tokens_to_capacity(new_capacity)?;

        tracing::info!(
            device = self.device,
            measured_util = %format!("{:.1}%", measured * 100.0),
            smoothed_util = %format!("{:.1}%", smoothed * 100.0),
            target_util = %format!("{:.1}%", self.cfg.target_utilization * 100.0),
            error = %format!("{:.1}%", error * 100.0),
            drain_rate = %format!("{:.1}/s", drain_rate),
            base_rate = %format!("{:.1}/s", base_rate),
            correction = %format!("{:.1}%", correction * 100.0),
            new_rate = %format!("{:.1}/s", new_rate),
            new_capacity = %format!("{:.1}", new_capacity),
            "ERL controller update"
        );

        Ok(DeviceControllerState {
            target_utilization: self.cfg.target_utilization,
            smoothed_utilization: smoothed,
            current_rate: new_rate,
            current_capacity: new_capacity,
            token_drain_rate: drain_rate,
        })
    }

    /// Apply exponential moving average to smooth utilization measurements.
    fn smooth_utilization(&mut self, measured: f64) -> f64 {
        let alpha = self.cfg.filter_alpha;
        let smoothed = match self.smoothed_util {
            Some(prev) => alpha * measured + (1.0 - alpha) * prev,
            None => measured,
        };
        self.smoothed_util = Some(smoothed);
        smoothed
    }

    /// Estimate token drain rate from bucket level changes.
    fn estimate_drain_rate(&mut self, delta_time: f64) -> Result<f64, ErlError> {
        let current_tokens = self.backend.read_token_state(self.device)?.tokens;

        // Expected tokens = last level + refill during delta_time
        let expected_tokens = self.last_token_level + self.current_rate * delta_time;

        // Actual drain = expected - actual
        let drain_rate = ((expected_tokens - current_tokens) / delta_time).max(0.0);

        self.last_token_level = current_tokens;

        Ok(drain_rate)
    }

    /// Calculate base refill rate from current utilization and drain rate.
    ///
    /// The idea: if we're at `actual_util` with `drain_rate`, then to reach
    /// `target_util` we need: `base_rate = drain_rate × (target / actual)`
    fn calculate_base_rate(&self, smoothed_util: f64, drain_rate: f64) -> f64 {
        if smoothed_util > 0.01 {
            // Theoretical base rate to reach target
            let theoretical = drain_rate * (self.cfg.target_utilization / smoothed_util);
            theoretical.clamp(self.cfg.rate_min, self.cfg.rate_max)
        } else {
            // Very low utilization - maintain current rate or use minimum
            self.current_rate.max(self.cfg.rate_min)
        }
    }

    /// Compute PID correction term.
    ///
    /// Returns a correction factor in the range [-0.5, 0.5] to apply to base_rate.
    fn compute_pid_correction(&mut self, error: f64, delta_time: f64) -> f64 {
        // Proportional term
        let p = self.cfg.kp * error;

        // Integral term with anti-windup
        self.integral += error * delta_time;
        self.integral = self.integral.clamp(-1.0, 1.0);
        let i = self.cfg.ki * self.integral;

        // Derivative term
        let derivative = (error - self.last_error) / delta_time;
        let d = self.cfg.kd * derivative;
        self.last_error = error;

        // Total correction, clamped to avoid over-reaction
        (p + i + d).clamp(-0.5, 0.5)
    }

    /// Clamp tokens to capacity if they exceed it.
    fn clamp_tokens_to_capacity(&self, capacity: f64) -> Result<(), ErlError> {
        let mut state = self.backend.read_token_state(self.device)?;
        if state.tokens > capacity {
            state.tokens = capacity;
            self.backend.write_token_state(self.device, state)?;
        }
        Ok(())
    }

    /// Update with explicit delta time.
    pub fn update(
        &mut self,
        utilization: f64,
        delta_time: f64,
    ) -> Result<DeviceControllerState, ErlError> {
        if delta_time < self.cfg.min_delta_time {
            return Ok(self.state());
        }
        self.update_internal(utilization, delta_time)
    }

    /// Update with timestamp (calculates delta automatically).
    pub fn update_with_timestamp(
        &mut self,
        utilization: f64,
        timestamp_micros: u64,
    ) -> Result<DeviceControllerState, ErlError> {
        let seconds = timestamp_micros as f64 / 1_000_000.0;
        let delta = if let Some(prev) = self.last_timestamp {
            let raw_delta = seconds - prev;
            if raw_delta < self.cfg.min_delta_time {
                return Ok(self.state());
            }
            raw_delta
        } else {
            self.cfg.min_delta_time
        };
        self.last_timestamp = Some(seconds);
        self.update_internal(utilization, delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DeviceBackend, DeviceQuota, TokenState};
    use core::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug)]
    struct MockBackend {
        quota_capacity: AtomicU64,
        quota_refill_rate: AtomicU64,
        tokens: AtomicU64,
        last_update: AtomicU64,
    }

    impl MockBackend {
        fn new(capacity: f64, refill_rate: f64, tokens: f64) -> Self {
            Self {
                quota_capacity: AtomicU64::new(capacity.to_bits()),
                quota_refill_rate: AtomicU64::new(refill_rate.to_bits()),
                tokens: AtomicU64::new(tokens.to_bits()),
                last_update: AtomicU64::new(0),
            }
        }
    }

    impl DeviceBackend for MockBackend {
        fn read_token_state(&self, _device: usize) -> Result<TokenState, ErlError> {
            Ok(TokenState::new(
                f64::from_bits(self.tokens.load(Ordering::Relaxed)),
                f64::from_bits(self.last_update.load(Ordering::Relaxed)),
            ))
        }

        fn write_token_state(&self, _device: usize, state: TokenState) -> Result<(), ErlError> {
            self.tokens.store(state.tokens.to_bits(), Ordering::Relaxed);
            self.last_update
                .store(state.last_update.to_bits(), Ordering::Relaxed);
            Ok(())
        }

        fn read_quota(&self, _device: usize) -> Result<DeviceQuota, ErlError> {
            Ok(DeviceQuota::new(
                f64::from_bits(self.quota_capacity.load(Ordering::Relaxed)),
                f64::from_bits(self.quota_refill_rate.load(Ordering::Relaxed)),
            ))
        }

        fn write_refill_rate(&self, _device: usize, refill_rate: f64) -> Result<(), ErlError> {
            self.quota_refill_rate
                .store(refill_rate.to_bits(), Ordering::Relaxed);
            Ok(())
        }

        fn write_capacity(&self, _device: usize, capacity: f64) -> Result<(), ErlError> {
            self.quota_capacity
                .store(capacity.to_bits(), Ordering::Relaxed);
            Ok(())
        }

        fn fetch_sub_tokens(&self, _device: usize, cost: f64) -> Result<f64, ErlError> {
            let capacity = f64::from_bits(self.quota_capacity.load(Ordering::Relaxed));
            loop {
                let current_bits = self.tokens.load(Ordering::Relaxed);
                let current = f64::from_bits(current_bits);
                if current < cost {
                    return Ok(current);
                }
                let new_tokens = (current - cost).max(0.0).min(capacity);
                if self
                    .tokens
                    .compare_exchange(
                        current_bits,
                        new_tokens.to_bits(),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    return Ok(current);
                }
            }
        }

        fn fetch_add_tokens(&self, _device: usize, amount: f64) -> Result<f64, ErlError> {
            let capacity = f64::from_bits(self.quota_capacity.load(Ordering::Relaxed));
            loop {
                let current_bits = self.tokens.load(Ordering::Relaxed);
                let current = f64::from_bits(current_bits);
                let new_tokens = (current + amount).min(capacity).max(0.0);
                if self
                    .tokens
                    .compare_exchange(
                        current_bits,
                        new_tokens.to_bits(),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    return Ok(current);
                }
            }
        }
    }

    #[test]
    fn controller_initializes_correctly() {
        let backend = MockBackend::new(0.0, 0.0, 0.0);
        let cfg = DeviceControllerConfig::new(0.7, 50000.0);
        let ctrl = DeviceController::new(backend, 0, cfg).unwrap();

        assert_eq!(ctrl.cfg.target_utilization, 0.7);
        assert!(ctrl.current_rate >= ctrl.cfg.rate_min);
        assert!(ctrl.current_rate <= ctrl.cfg.rate_max);
    }

    #[test]
    fn controller_increases_rate_when_under_target() {
        let backend = MockBackend::new(1000.0, 100.0, 500.0);
        let cfg = DeviceControllerConfig::new(0.7, 50000.0);
        let mut ctrl = DeviceController::new(backend, 0, cfg).unwrap();

        let rate_before = ctrl.current_rate;

        // Utilization 20% when target is 70% -> should increase rate
        ctrl.update(0.2, 0.1).unwrap();

        let rate_after = ctrl.current_rate;
        assert!(
            rate_after > rate_before,
            "Rate should increase when utilization is below target"
        );
    }

    #[test]
    fn controller_decreases_rate_when_over_target() {
        let backend = MockBackend::new(1000.0, 100.0, 500.0);
        let cfg = DeviceControllerConfig::new(0.5, 50000.0);
        let mut ctrl = DeviceController::new(backend, 0, cfg).unwrap();

        // First establish a higher rate
        ctrl.update(0.3, 0.1).unwrap();
        ctrl.update(0.3, 0.1).unwrap();
        let rate_before = ctrl.current_rate;

        // Now push utilization above target
        ctrl.update(0.95, 0.1).unwrap();

        let rate_after = ctrl.current_rate;
        assert!(
            rate_after < rate_before,
            "Rate should decrease when utilization is above target"
        );
    }

    #[test]
    fn controller_respects_rate_limits() {
        let backend = MockBackend::new(1000.0, 100.0, 500.0);
        let cfg = DeviceControllerConfig {
            rate_min: 50.0,
            rate_max: 500.0,
            ..DeviceControllerConfig::new(0.5, 500.0)
        };
        let mut ctrl = DeviceController::new(backend, 0, cfg).unwrap();

        // Try to push rate very low
        for _ in 0..10 {
            ctrl.update(0.99, 0.1).unwrap();
        }
        assert!(
            ctrl.current_rate >= 50.0,
            "Rate should not go below rate_min"
        );

        // Try to push rate very high
        for _ in 0..10 {
            ctrl.update(0.01, 0.1).unwrap();
        }
        assert!(
            ctrl.current_rate <= 500.0,
            "Rate should not exceed rate_max"
        );
    }

    #[test]
    fn controller_smooths_utilization() {
        let backend = MockBackend::new(1000.0, 100.0, 500.0);
        let cfg = DeviceControllerConfig {
            filter_alpha: 0.3,
            ..DeviceControllerConfig::new(0.5, 50000.0)
        };
        let mut ctrl = DeviceController::new(backend, 0, cfg).unwrap();

        // Feed alternating utilization values
        ctrl.update(0.8, 0.1).unwrap();

        ctrl.update(0.2, 0.1).unwrap();
        let state2 = ctrl.state();

        // Smoothed value should be between the extremes
        assert!(
            state2.smoothed_utilization > 0.2 && state2.smoothed_utilization < 0.8,
            "Smoothed utilization should filter noise"
        );
    }

    #[test]
    fn controller_handles_zero_utilization() {
        let backend = MockBackend::new(1000.0, 100.0, 500.0);
        let cfg = DeviceControllerConfig::new(0.5, 50000.0);
        let mut ctrl = DeviceController::new(backend, 0, cfg).unwrap();

        // Feed zero utilization repeatedly
        for _ in 0..5 {
            let result = ctrl.update(0.0, 0.1);
            assert!(result.is_ok(), "Controller should handle zero utilization");
        }

        // Rate should still be above minimum
        assert!(
            ctrl.current_rate >= ctrl.cfg.rate_min,
            "Rate should never drop below rate_min"
        );
    }

    #[test]
    fn capacity_scales_with_rate() {
        let backend = MockBackend::new(1000.0, 100.0, 500.0);
        let cfg = DeviceControllerConfig::new(0.5, 50000.0);
        let mut ctrl = DeviceController::new(backend, 0, cfg).unwrap();

        ctrl.update(0.2, 0.1).unwrap();
        let state1 = ctrl.state();

        // Continue to increase rate
        for _ in 0..5 {
            ctrl.update(0.2, 0.1).unwrap();
        }
        let state2 = ctrl.state();

        if state2.current_rate > state1.current_rate {
            assert!(
                state2.current_capacity >= state1.current_capacity,
                "Capacity should scale with rate"
            );
        }
    }

    #[test]
    fn controller_with_timestamp_updates() {
        let backend = MockBackend::new(1000.0, 100.0, 500.0);
        let cfg = DeviceControllerConfig::new(0.5, 50000.0);
        let mut ctrl = DeviceController::new(backend, 0, cfg).unwrap();

        // Update with timestamps (in microseconds)
        let t1 = 1_000_000; // 1 second
        let t2 = 1_200_000; // 1.2 seconds (0.2s delta)

        ctrl.update_with_timestamp(0.3, t1).unwrap();
        let result = ctrl.update_with_timestamp(0.4, t2);
        assert!(result.is_ok(), "Timestamp-based updates should work");
    }
}
