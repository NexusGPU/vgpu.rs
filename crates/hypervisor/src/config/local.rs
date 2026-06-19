use std::path::PathBuf;

use clap::Args;
use erl::DeviceControllerConfig;

use crate::cmd::local::LocalConfig;

/// Arguments for local testing mode
#[derive(Args, Debug, Clone)]
pub struct LocalArgs {
    /// GPU device indices to manage (comma-separated)
    #[arg(short, long, value_delimiter = ',', required = true)]
    pub devices: Vec<usize>,

    /// Target GPU utilization (0.0 to 1.0)
    #[arg(short, long, default_value = "0.5")]
    pub target_util: f64,

    /// Shared memory directory path
    #[arg(long, default_value = "/tmp/tensor-fusion-local")]
    pub shm_path: PathBuf,

    /// HTTP API server port
    #[arg(short = 'p', long, default_value = "8080")]
    pub api_port: u16,

    /// Enable compute shard mode
    #[arg(long, default_value = "false")]
    pub compute_shard: bool,

    /// Isolation level (soft/hard/none)
    #[arg(long)]
    pub isolation: Option<String>,

    /// Controller update interval in milliseconds
    #[arg(long, default_value = "100")]
    pub update_interval_ms: u64,

    /// Initial token refill rate (tokens/sec)
    #[arg(long, default_value = "50.0")]
    pub initial_rate: f64,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,
}

impl From<LocalArgs> for LocalConfig {
    fn from(args: LocalArgs) -> Self {
        let controller_config = DeviceControllerConfig {
            target_utilization: args.target_util,
            ..DeviceControllerConfig::default()
        };

        Self {
            devices: args.devices,
            target_utilization: args.target_util,
            shm_path: args.shm_path,
            api_port: args.api_port,
            compute_shard: args.compute_shard,
            isolation: args.isolation,
            update_interval_ms: args.update_interval_ms,
            initial_rate: args.initial_rate,
            controller_config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_args_to_config_conversion() {
        let args = LocalArgs {
            devices: vec![0, 1],
            target_util: 0.8,
            shm_path: PathBuf::from("/tmp/test-shm"),
            api_port: 9090,
            compute_shard: true,
            isolation: Some("soft".to_string()),
            update_interval_ms: 200,
            initial_rate: 100.0,
            verbose: false,
        };

        let config: LocalConfig = args.into();

        assert_eq!(config.devices, vec![0, 1]);
        assert_eq!(config.target_utilization, 0.8);
        assert_eq!(config.shm_path, PathBuf::from("/tmp/test-shm"));
        assert_eq!(config.api_port, 9090);
        assert!(config.compute_shard);
        assert_eq!(config.isolation, Some("soft".to_string()));
        assert_eq!(config.update_interval_ms, 200);
        assert_eq!(config.initial_rate, 100.0);
        assert_eq!(config.controller_config.target_utilization, 0.8);
    }

    #[test]
    fn test_local_args_defaults() {
        let args = LocalArgs {
            devices: vec![0],
            target_util: 0.5,
            shm_path: PathBuf::from("/tmp/tensor-fusion-local"),
            api_port: 8080,
            compute_shard: false,
            isolation: None,
            update_interval_ms: 100,
            initial_rate: 50.0,
            verbose: false,
        };

        let config: LocalConfig = args.into();

        // Verify default values
        assert_eq!(config.target_utilization, 0.5);
        assert_eq!(config.api_port, 8080);
        assert_eq!(config.update_interval_ms, 100);
        assert_eq!(config.initial_rate, 50.0);
        assert!(!config.compute_shard);
        assert_eq!(config.isolation, None);
    }

    #[test]
    fn test_controller_config_target_utilization_sync() {
        let args = LocalArgs {
            devices: vec![0],
            target_util: 0.65,
            shm_path: PathBuf::from("/tmp/test"),
            api_port: 8080,
            compute_shard: false,
            isolation: None,
            update_interval_ms: 100,
            initial_rate: 50.0,
            verbose: false,
        };

        let config: LocalConfig = args.into();

        // Ensure target_utilization is synced to controller_config
        assert_eq!(config.target_utilization, 0.65);
        assert_eq!(config.controller_config.target_utilization, 0.65);
    }
}
