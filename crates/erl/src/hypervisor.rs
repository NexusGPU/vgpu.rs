use crate::backend::{DeviceBackend, TokenState};
use crate::{ErlError, PidConfig, PidController};
use error_stack::Result;

/// PID-based controller configuration for a single GPU.
#[derive(Debug, Clone)]
pub struct DeviceControllerConfig {
    pub target_utilization: f64,
    pub burst_seconds: f64,
    pub capacity_floor: f64,
    pub pid: PidConfig,
}

impl Default for DeviceControllerConfig {
    fn default() -> Self {
        Self {
            target_utilization: 0.5,
            burst_seconds: 0.25,
            capacity_floor: 1.0,
            pid: PidConfig::default(),
        }
    }
}

/// Snapshot of the controller after the last update.
#[derive(Debug, Clone)]
pub struct DeviceControllerState {
    pub target_utilization: f64,
    pub last_utilization: f64,
    pub last_refill_rate: f64,
    pub last_token_level: f64,
    pub token_drain_rate: f64,
}

/// Hypervisor-side controller that updates refill rates based on utilization samples.
#[derive(Debug)]
pub struct DeviceController<B: DeviceBackend> {
    backend: B,
    device: usize,
    cfg: DeviceControllerConfig,
    pid: PidController,
    state: DeviceControllerState,
    last_timestamp: Option<f64>,
}

impl<B: DeviceBackend> DeviceController<B> {
    pub fn new(
        backend: B,
        device: usize,
        mut cfg: DeviceControllerConfig,
    ) -> Result<Self, ErlError> {
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

        let quota = backend.read_quota(device)?;
        let start_rate = quota
            .refill_rate
            .clamp(cfg.pid.output_min, cfg.pid.output_max);
        cfg.pid.initial_output = start_rate;
        let mut pid = PidController::new(cfg.pid.clone())?;
        pid.reset(Some(start_rate));

        let initial_capacity = if cfg.burst_seconds > 0.0 {
            (start_rate * cfg.burst_seconds).max(cfg.capacity_floor)
        } else {
            quota.capacity
        };
        if cfg.burst_seconds > 0.0 {
            backend.write_capacity(device, initial_capacity)?;
            clamp_tokens_if_needed(&backend, device, initial_capacity)?;
        }

        Ok(Self {
            backend,
            device,
            cfg: DeviceControllerConfig {
                pid: cfg.pid.clone(),
                ..cfg
            },
            pid,
            state: DeviceControllerState {
                target_utilization: cfg.target_utilization,
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

    /// Internal update method that performs PID control and token refilling.
    ///
    /// This method:
    /// 1. Runs PID controller to compute new refill_rate
    /// 2. Adds tokens based on new_rate × delta_time (with saturation detection)
    /// 3. Updates capacity and clamps tokens if needed
    ///
    /// # Convergence guarantees
    ///
    /// - When tokens are saturated (at capacity), we detect it and adjust PID behavior
    /// - This prevents runaway rate increases when there's no load
    /// - The PID integral term is bounded to prevent windup
    fn update_internal(
        &mut self,
        utilization: f64,
        delta_time: f64,
    ) -> Result<DeviceControllerState, ErlError> {
        let measured = utilization.clamp(0.0, 1.0);

        // Check token saturation BEFORE PID update
        let quota = self.backend.read_quota(self.device)?;
        let current_state = self.backend.read_token_state(self.device)?;
        let token_saturation_ratio = if quota.capacity > 0.0 {
            current_state.tokens / quota.capacity
        } else {
            0.0
        };

        // Calculate token consumption rate for logging/debugging
        let expected_tokens =
            self.state.last_token_level + self.state.last_refill_rate * delta_time;
        let actual_tokens = current_state.tokens;
        let token_drain_rate = (expected_tokens - actual_tokens) / delta_time; // tokens/sec consumed

        // If tokens are highly saturated (>95%) AND utilization is very low (<5%),
        // treat this as "no demand" scenario - don't let PID increase rate further
        let effective_utilization = if token_saturation_ratio > 0.95 && measured < 0.05 {
            // Clamp measured utilization to target to prevent PID from increasing rate
            // This creates a "soft ceiling" that prevents runaway in no-load scenarios
            tracing::debug!(
                device = self.device,
                tokens = current_state.tokens,
                capacity = quota.capacity,
                saturation = %format!("{:.1}%", token_saturation_ratio * 100.0),
                measured_utilization = %format!("{:.1}%", measured * 100.0),
                "ERL: Token bucket saturated with low utilization - clamping to target to prevent runaway"
            );
            self.cfg.target_utilization
        } else {
            measured
        };

        let old_rate = self.state.last_refill_rate;
        let new_rate = self.pid.update(
            self.cfg.target_utilization,
            effective_utilization,
            delta_time,
        )?;

        // Refill tokens using the NEW rate (PID-adjusted amount)
        let token_amount = new_rate * delta_time;
        let tokens_before = if token_amount > 0.0 {
            self.backend.fetch_add_tokens(self.device, token_amount)?
        } else {
            current_state.tokens
        };

        let tokens_after = self.backend.read_token_state(self.device)?.tokens;

        // Detailed logging for debugging
        tracing::debug!(
            device = self.device,
            measured_util = %format!("{:.1}%", measured * 100.0),
            effective_util = %format!("{:.1}%", effective_utilization * 100.0),
            target_util = %format!("{:.1}%", self.cfg.target_utilization * 100.0),
            old_rate = %format!("{:.1}", old_rate),
            new_rate = %format!("{:.1}", new_rate),
            delta_time = %format!("{:.3}s", delta_time),
            token_add = %format!("{:.1}", token_amount),
            token_drain = %format!("{:.1}/s", token_drain_rate),
            tokens_before = %format!("{:.1}", tokens_before),
            tokens_after = %format!("{:.1}", tokens_after),
            capacity = %format!("{:.1}", quota.capacity),
            pid_error = %format!("{:.1}%", (self.cfg.target_utilization - effective_utilization) * 100.0),
            "ERL: Controller update"
        );

        // Update shared memory with new rate and capacity
        self.backend.write_refill_rate(self.device, new_rate)?;

        if self.cfg.burst_seconds > 0.0 {
            let new_capacity = (new_rate * self.cfg.burst_seconds).max(self.cfg.capacity_floor);
            self.backend.write_capacity(self.device, new_capacity)?;
            clamp_tokens_if_needed(&self.backend, self.device, new_capacity)?;
        }

        self.state.last_utilization = measured;
        self.state.last_refill_rate = new_rate;
        self.state.last_token_level = tokens_after;
        self.state.token_drain_rate = token_drain_rate;
        Ok(self.state.clone())
    }

    /// Update controller with utilization measurement and explicit delta time.
    ///
    /// For testing or scenarios where you already know the time delta.
    pub fn update(
        &mut self,
        utilization: f64,
        delta_time: f64,
    ) -> Result<DeviceControllerState, ErlError> {
        self.update_internal(utilization, delta_time)
    }

    /// Update controller with utilization measurement and timestamp.
    ///
    /// Automatically calculates delta_time from the last update.
    pub fn update_with_timestamp(
        &mut self,
        utilization: f64,
        timestamp_micros: u64,
    ) -> Result<DeviceControllerState, ErlError> {
        let seconds = timestamp_micros as f64 / 1_000_000.0;
        let delta = if let Some(prev) = self.last_timestamp {
            (seconds - prev).max(self.cfg.pid.min_delta_time)
        } else {
            self.cfg.pid.min_delta_time
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
    use std::sync::Mutex;

    #[derive(Debug)]
    struct MockBackend {
        quota: Mutex<DeviceQuota>,
        tokens: Mutex<TokenState>,
    }

    impl MockBackend {
        fn new(capacity: f64, refill_rate: f64) -> Self {
            Self {
                quota: Mutex::new(DeviceQuota::new(capacity, refill_rate)),
                tokens: Mutex::new(TokenState::new(refill_rate, 0.0)),
            }
        }
    }

    impl DeviceBackend for MockBackend {
        fn read_token_state(&self, _device: usize) -> Result<TokenState, ErlError> {
            Ok(*self.tokens.lock().unwrap())
        }

        fn write_token_state(&self, _device: usize, state: TokenState) -> Result<(), ErlError> {
            *self.tokens.lock().unwrap() = state;
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
            let mut tokens = self.tokens.lock().unwrap();
            let before = tokens.tokens;
            tokens.tokens = (tokens.tokens - cost).max(0.0);
            Ok(before)
        }

        fn fetch_add_tokens(&self, _device: usize, amount: f64) -> Result<f64, ErlError> {
            let mut tokens = self.tokens.lock().unwrap();
            let capacity = self.quota.lock().unwrap().capacity;
            let before = tokens.tokens;
            tokens.tokens = (tokens.tokens + amount).min(capacity).max(0.0);
            Ok(before)
        }
    }

    #[test]
    fn pid_increases_rate_when_under_target() {
        let backend = MockBackend::new(10.0, 5.0);
        let cfg = DeviceControllerConfig {
            target_utilization: 0.7,
            burst_seconds: 0.2,
            capacity_floor: 1.0,
            pid: PidConfig {
                tuning: crate::PidTuning::new(2.0, 0.5, 0.0),
                output_min: 0.5,
                output_max: 50.0,
                initial_output: 5.0,
                integral_limit: 100.0,
                derivative_filter: 0.0,
                min_delta_time: 1e-3,
            },
        };
        let mut ctrl = DeviceController::new(backend, 0, cfg).unwrap();
        let before = ctrl.state().last_refill_rate;
        let after = ctrl.update(0.2, 0.1).unwrap().last_refill_rate;
        assert!(after > before);
    }
}
