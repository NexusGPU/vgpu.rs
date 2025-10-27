use crate::ErlError;
use error_stack::Result;

/// Core PID gains.
#[derive(Debug, Clone, Copy)]
pub struct PidTuning {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
}

impl PidTuning {
    pub const fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self { kp, ki, kd }
    }
}

/// Configuration for a PID controller.
#[derive(Debug, Clone)]
pub struct PidConfig {
    pub tuning: PidTuning,
    pub output_min: f64,
    pub output_max: f64,
    pub initial_output: f64,
    pub integral_limit: f64,
    /// Exponential smoothing factor for the derivative term. 0.0 disables smoothing.
    pub derivative_filter: f64,
    /// Minimum delta time in seconds to guard against division by zero.
    pub min_delta_time: f64,
}

impl Default for PidConfig {
    fn default() -> Self {
        Self {
            tuning: PidTuning::new(0.8, 0.3, 0.0),
            output_min: 0.0,
            output_max: 10_000.0,
            initial_output: 10.0,
            integral_limit: 1_000.0,
            derivative_filter: 0.0,
            min_delta_time: 1e-3,
        }
    }
}

/// Simple PID controller with clamped output and integral anti-windup.
#[derive(Debug, Clone)]
pub struct PidController {
    cfg: PidConfig,
    integral: f64,
    last_error: f64,
    last_derivative: f64,
    output: f64,
    initialized: bool,
}

impl PidController {
    pub fn new(cfg: PidConfig) -> Result<Self, ErlError> {
        if cfg.output_max <= cfg.output_min {
            return Err(error_stack::report!(ErlError::invalid_config(
                "output_max must be greater than output_min"
            )));
        }
        if !cfg.integral_limit.is_finite() || cfg.integral_limit <= 0.0 {
            return Err(error_stack::report!(ErlError::invalid_config(
                "integral_limit must be a positive finite value"
            )));
        }
        let initial = cfg.initial_output.clamp(cfg.output_min, cfg.output_max);

        Ok(Self {
            cfg,
            integral: 0.0,
            last_error: 0.0,
            last_derivative: 0.0,
            output: initial,
            initialized: false,
        })
    }

    pub fn output(&self) -> f64 {
        self.output
    }

    pub fn reset(&mut self, new_output: Option<f64>) {
        self.integral = 0.0;
        self.last_error = 0.0;
        self.last_derivative = 0.0;
        self.initialized = false;
        if let Some(out) = new_output {
            self.output = out.clamp(self.cfg.output_min, self.cfg.output_max);
        } else {
            self.output = self
                .cfg
                .initial_output
                .clamp(self.cfg.output_min, self.cfg.output_max);
        }
    }

    /// Update controller with the latest measurement and timestep.
    pub fn update(
        &mut self,
        setpoint: f64,
        measurement: f64,
        delta_time: f64,
    ) -> Result<f64, ErlError> {
        let dt = delta_time.max(self.cfg.min_delta_time);
        let error = setpoint - measurement;

        if !self.initialized {
            self.last_error = error;
            self.initialized = true;
        }

        let mut next_integral = self.integral + error * dt;
        next_integral = next_integral.clamp(-self.cfg.integral_limit, self.cfg.integral_limit);

        let raw_derivative = (error - self.last_error) / dt;
        let filter = self.cfg.derivative_filter.clamp(0.0, 1.0);
        let derivative = if filter == 0.0 {
            raw_derivative
        } else {
            let filtered = filter * raw_derivative + (1.0 - filter) * self.last_derivative;
            self.last_derivative = filtered;
            filtered
        };

        let mut delta = self.cfg.tuning.kp * error
            + self.cfg.tuning.ki * next_integral
            + self.cfg.tuning.kd * derivative;

        let mut candidate = self.output + delta;
        candidate = candidate.clamp(self.cfg.output_min, self.cfg.output_max);

        // Simple anti-windup: if saturated and error pushes further, keep previous integral.
        let saturating_high = (candidate - self.cfg.output_max).abs() < f64::EPSILON && delta > 0.0;
        let saturating_low = (candidate - self.cfg.output_min).abs() < f64::EPSILON && delta < 0.0;

        if saturating_high || saturating_low {
            next_integral = self.integral;
            delta = self.cfg.tuning.kp * error
                + self.cfg.tuning.ki * next_integral
                + self.cfg.tuning.kd * derivative;
            candidate = (self.output + delta).clamp(self.cfg.output_min, self.cfg.output_max);
        }

        self.integral = next_integral;
        self.output = candidate;
        self.last_error = error;
        if filter == 0.0 {
            self.last_derivative = raw_derivative;
        }

        Ok(self.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_respects_output_bounds() {
        let cfg = PidConfig {
            tuning: PidTuning::new(2.0, 0.0, 0.0),
            output_min: 0.0,
            output_max: 1.0,
            initial_output: 0.5,
            integral_limit: 10.0,
            derivative_filter: 0.0,
            min_delta_time: 1e-3,
        };
        let mut pid = PidController::new(cfg).unwrap();

        for _ in 0..10 {
            let out = pid.update(1.0, 0.0, 0.1).unwrap();
            assert!(out <= 1.0 + 1e-9);
        }
        for _ in 0..10 {
            let out = pid.update(0.0, 1.0, 0.1).unwrap();
            assert!(out >= -1e-9);
        }
    }
}
