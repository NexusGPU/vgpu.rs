use crate::ErlError;
use crate::backend::{DeviceBackend, TokenState};
use error_stack::Result;

/// Configuration for the PID-based device controller.
#[derive(Debug, Clone)]
pub struct DeviceControllerConfig {
    /// Target GPU utilization (0.0 to 1.0, e.g., 0.7 = 70%)
    pub target_utilization: f64,
    /// Burst capacity in seconds (capacity = refill_rate * burst_seconds)
    pub burst_seconds: f64,
    /// Minimum capacity floor in tokens
    pub capacity_floor: f64,
    /// PID proportional gain
    pub kp: f64,
    /// PID integral gain
    pub ki: f64,
    /// PID derivative gain
    pub kd: f64,
    /// Maximum limit for integral term
    pub integral_limit: f64,
    /// Minimum refill rate (tokens/second)
    pub rate_min: f64,
    /// Maximum refill rate (tokens/second)
    pub rate_max: f64,
    /// Initial refill rate (tokens/second)
    pub rate_initial: f64,
}

impl Default for DeviceControllerConfig {
    fn default() -> Self {
        Self {
            target_utilization: 0.5,
            burst_seconds: 0.25,
            capacity_floor: 1.0,
            kp: 0.8,
            ki: 0.3,
            kd: 0.0,
            integral_limit: 1_000.0,
            rate_min: 0.0,
            rate_max: 10_000.0,
            rate_initial: 10.0,
        }
    }
}

/// Snapshot of controller state after an update.
#[derive(Debug, Clone)]
pub struct DeviceControllerState {
    pub target_utilization: f64,
    pub last_utilization: f64,
    pub last_refill_rate: f64,
    pub last_token_level: f64,
    pub token_drain_rate: f64,
}

/// PID-based controller that dynamically adjusts token refill rates.
///
/// The controller receives utilization measurements and uses PID control
/// to compute appropriate refill rates, implementing elastic rate limiting.
#[derive(Debug)]
pub struct DeviceController<B: DeviceBackend> {
    backend: B,
    device: usize,
    cfg: DeviceControllerConfig,
    pid: pid::Pid<f64>,
    state: DeviceControllerState,
    current_rate: f64,
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
        if cfg.burst_seconds < 0.0 {
            return Err(error_stack::report!(ErlError::invalid_config(
                "burst_seconds must be non-negative"
            )));
        }
        if cfg.capacity_floor < 0.0 {
            return Err(error_stack::report!(ErlError::invalid_config(
                "capacity_floor must be non-negative"
            )));
        }
        if cfg.rate_max <= cfg.rate_min {
            return Err(error_stack::report!(ErlError::invalid_config(
                "rate_max must be greater than rate_min"
            )));
        }

        // Initialize PID controller
        let mut pid = pid::Pid::new(cfg.target_utilization, cfg.rate_max);
        pid.p(cfg.kp, cfg.rate_max);
        pid.i(cfg.ki, cfg.integral_limit);
        pid.d(cfg.kd, cfg.rate_max);

        // Get current quota and clamp initial rate
        let quota = backend.read_quota(device)?;
        let start_rate = quota.refill_rate.clamp(cfg.rate_min, cfg.rate_max);

        // Calculate initial capacity
        let initial_capacity = if cfg.burst_seconds > 0.0 {
            (start_rate * cfg.burst_seconds).max(cfg.capacity_floor)
        } else {
            quota.capacity
        };

        // Set initial capacity in backend
        if cfg.burst_seconds > 0.0 {
            backend.write_capacity(device, initial_capacity)?;
            clamp_tokens_if_needed(&backend, device, initial_capacity)?;
        }

        let target_utilization = cfg.target_utilization;

        Ok(Self {
            backend,
            device,
            cfg,
            pid,
            current_rate: start_rate,
            state: DeviceControllerState {
                target_utilization,
                last_utilization: 0.0,
                last_refill_rate: start_rate,
                last_token_level: initial_capacity,
                token_drain_rate: 0.0,
            },
            last_timestamp: None,
        })
    }

    pub fn state(&self) -> &DeviceControllerState {
        &self.state
    }

    /// Update controller with new utilization measurement.
    fn update_internal(
        &mut self,
        utilization: f64,
        delta_time: f64,
    ) -> Result<DeviceControllerState, ErlError> {
        let measured = utilization.clamp(0.0, 1.0);

        // Check token saturation before PID update
        let quota = self.backend.read_quota(self.device)?;
        let current_state = self.backend.read_token_state(self.device)?;
        let token_saturation_ratio = if quota.capacity > 0.0 {
            current_state.tokens / quota.capacity
        } else {
            0.0
        };

        // Calculate token drain rate
        let expected_tokens =
            self.state.last_token_level + self.state.last_refill_rate * delta_time;
        let actual_tokens = current_state.tokens;
        let token_drain_rate = (expected_tokens - actual_tokens) / delta_time;

        // Apply saturation clamping ONLY when there's genuinely no workload:
        // 1. Token bucket is saturated (>95% full)
        // 2. GPU utilization is very low (<5%)
        // 3. Token drain rate is negligible (< 10% of min refill rate)
        // This prevents clamping when tokens are exhausted due to high demand
        let is_truly_idle = token_saturation_ratio > 0.95
            && measured < 0.05
            && token_drain_rate < self.cfg.rate_min * 0.1;

        let effective_utilization = if is_truly_idle {
            tracing::debug!(
                device = self.device,
                tokens = current_state.tokens,
                capacity = quota.capacity,
                saturation = %format!("{:.1}%", token_saturation_ratio * 100.0),
                measured_utilization = %format!("{:.1}%", measured * 100.0),
                token_drain_rate = %format!("{:.1}/s", token_drain_rate),
                "Token bucket saturated with no workload - clamping to prevent runaway"
            );
            self.cfg.target_utilization
        } else {
            measured
        };

        // Update PID controller
        let control_output = self.pid.next_control_output(effective_utilization);

        // Apply control signal incrementally
        let delta = control_output.output;
        let new_rate = (self.current_rate + delta).clamp(self.cfg.rate_min, self.cfg.rate_max);
        self.current_rate = new_rate;

        // Refill tokens
        let token_amount = new_rate * delta_time;
        let tokens_before = if token_amount > 0.0 {
            self.backend.fetch_add_tokens(self.device, token_amount)?
        } else {
            current_state.tokens
        };

        let tokens_after = self.backend.read_token_state(self.device)?.tokens;

        tracing::debug!(
            device = self.device,
            measured_util = %format!("{:.1}%", measured * 100.0),
            effective_util = %format!("{:.1}%", effective_utilization * 100.0),
            target_util = %format!("{:.1}%", self.cfg.target_utilization * 100.0),
            old_rate = %format!("{:.1}", self.state.last_refill_rate),
            new_rate = %format!("{:.1}", new_rate),
            delta_time = %format!("{:.3}s", delta_time),
            token_add = %format!("{:.1}", token_amount),
            token_drain = %format!("{:.1}/s", token_drain_rate),
            tokens_before = %format!("{:.1}", tokens_before),
            tokens_after = %format!("{:.1}", tokens_after),
            capacity = %format!("{:.1}", quota.capacity),
            pid_p = %format!("{:.2}", control_output.p),
            pid_i = %format!("{:.2}", control_output.i),
            pid_d = %format!("{:.2}", control_output.d),
            "ERL controller update"
        );

        // Update backend with new rate and capacity
        self.backend.write_refill_rate(self.device, new_rate)?;

        if self.cfg.burst_seconds > 0.0 {
            let new_capacity = (new_rate * self.cfg.burst_seconds).max(self.cfg.capacity_floor);
            self.backend.write_capacity(self.device, new_capacity)?;
            clamp_tokens_if_needed(&self.backend, self.device, new_capacity)?;
        }

        // Update state
        self.state.last_utilization = measured;
        self.state.last_refill_rate = new_rate;
        self.state.last_token_level = tokens_after;
        self.state.token_drain_rate = token_drain_rate;

        Ok(self.state.clone())
    }

    /// Update with explicit delta time.
    pub fn update(
        &mut self,
        utilization: f64,
        delta_time: f64,
    ) -> Result<DeviceControllerState, ErlError> {
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
            (seconds - prev).max(1e-3)
        } else {
            1e-3
        };
        self.last_timestamp = Some(seconds);
        self.update_internal(utilization, delta)
    }
}

fn clamp_tokens_if_needed<B: DeviceBackend>(
    backend: &B,
    device: usize,
    capacity: f64,
) -> Result<(), ErlError> {
    let mut state = backend.read_token_state(device)?;
    if state.tokens > capacity {
        state.tokens = capacity;
        backend.write_token_state(device, TokenState::new(state.tokens, state.last_update))?;
    }
    Ok(())
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
        fn new(capacity: f64, refill_rate: f64) -> Self {
            Self {
                quota_capacity: AtomicU64::new(capacity.to_bits()),
                quota_refill_rate: AtomicU64::new(refill_rate.to_bits()),
                tokens: AtomicU64::new(refill_rate.to_bits()),
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
    fn controller_increases_rate_when_under_target() {
        let backend = MockBackend::new(10.0, 5.0);
        let cfg = DeviceControllerConfig {
            target_utilization: 0.7,
            burst_seconds: 0.2,
            capacity_floor: 1.0,
            kp: 2.0,
            ki: 0.5,
            kd: 0.0,
            integral_limit: 100.0,
            rate_min: 0.5,
            rate_max: 50.0,
            rate_initial: 5.0,
        };
        let mut ctrl = DeviceController::new(backend, 0, cfg).unwrap();
        let before = ctrl.state().last_refill_rate;

        // Utilization 20% when target is 70% -> should increase rate
        let after = ctrl.update(0.2, 0.1).unwrap().last_refill_rate;

        assert!(
            after > before,
            "Expected rate to increase: before={before}, after={after}"
        );
    }
}
