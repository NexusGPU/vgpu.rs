# Pod Management Module

The Pod Management module provides a comprehensive, type-safe, and highly efficient system for managing Kubernetes pods, containers, workers, and GPU resources in the TensorFusion hypervisor.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Pod Management Module                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    Types    │  │    Core     │  │      Services       │ │
│  │             │  │             │  │                     │ │
│  │ • Pod       │  │ • Registry  │  │ • DeviceService     │ │
│  │ • Worker    │  │ • Manager   │  │ • WorkerService     │ │
│  │ • Device    │  │ • Errors    │  │ • ResourceMonitor   │ │
│  │             │  │             │  │ • MetricsCollector  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 🔒 Type Safety
- **Strong Type System**: `PodId`, `WorkerId`, `ContainerId` prevent identifier confusion
- **State Management**: Explicit status enums for pods and workers
- **Compile-time Guarantees**: Rust's type system ensures correctness

### 🚀 Performance
- **Lock-free Operations**: Minimized contention with fine-grained locking
- **Async/Await**: Full async support for non-blocking operations
- **Efficient Data Structures**: Optimized HashMap usage and memory layout

### 🔧 Extensibility
- **Service Layer**: Pluggable services for different concerns
- **Metrics Integration**: Built-in Prometheus metrics support
- **Event System**: Comprehensive monitoring and alerting

### 🛡️ Reliability
- **Error Handling**: Comprehensive error types with context
- **Resource Management**: Automatic cleanup and leak prevention
- **Health Monitoring**: Continuous worker and resource health checks

## Core Components

### Types Module (`types/`)

Defines all data structures used throughout the pod management system.

#### Pod Types
```rust
use crate::domain::pod_management::types::*;

// Create a pod ID
let pod_id = PodId::new("my-namespace", "my-pod");

// Create a pod from worker info
let pod = Pod::new(worker_info, device_allocation);

// Check pod status
match pod.status {
    PodStatus::Running => println!("Pod is running"),
    PodStatus::Terminating => println!("Pod is terminating"),
    _ => {}
}
```

#### Worker Types
```rust
// Create a worker
let worker = Worker::new(
    host_pid,
    container_pid,
    "container-name".to_string(),
    QosLevel::High,
    worker_instance,
);

// Check worker health
if worker.is_active() {
    println!("Worker is healthy");
}
```

#### Device Types
```rust
// Create device allocation
let allocation = DeviceAllocation::new(device_configs);

// Check quotas
if let Some(quota) = allocation.get_quota(device_idx) {
    println!("CUDA limit: {}%", quota.cuda_limit);
}
```

### Core Module (`core/`)

Provides the fundamental registry and management functionality.

#### Pod Registry
```rust
use crate::domain::pod_management::core::PodRegistry;

let registry = PodRegistry::new();

// Register a pod
registry.register_pod(pod).await?;

// Find pods
let pod = registry.get_pod(&pod_id).await;
let pod_by_pid = registry.get_pod_by_pid(worker_pid).await;

// Add workers
registry.add_worker(&pod_id, "container1", host_pid, worker).await?;

// Get statistics
let stats = registry.stats().await;
println!("Total pods: {}, Active workers: {}", stats.total_pods, stats.total_workers);
```

#### Pod Manager
```rust
use crate::domain::pod_management::core::PodManager;

let manager = PodManager::new(
    host_pid_probe,
    command_dispatcher,
    hypervisor,
    nvml,
    device_service,
    worker_service,
    resource_monitor,
);

// Handle Kubernetes events
manager.handle_pod_created("my-pod", "my-namespace", worker_info, None).await?;
manager.handle_pod_deleted("my-pod", "my-namespace").await?;

// Initialize workers
let host_pid = manager.initialize_process(
    "my-pod",
    "my-namespace", 
    "container1",
    gpu_observer
).await?;
```

### Services Module (`services/`)

Provides specialized services for different aspects of pod management.

#### Device Service
```rust
use crate::domain::pod_management::services::DeviceService;

let device_service = DeviceService::new(nvml, shared_memory_manager, glob_pattern);

// Create device allocation
let allocation = device_service.create_allocation(&worker_info).await?;

// Register pod allocation
device_service.register_pod_allocation(&pod_id, &allocation)?;

// Monitor usage
let usage = device_service.get_device_usage(device_idx).await?;
let violations = device_service.check_resource_violations(&allocation).await?;
```

#### Worker Service
```rust
use crate::domain::pod_management::services::WorkerService;

let worker_service = WorkerService::new(
    host_pid_probe,
    command_dispatcher,
    hypervisor,
);

// Create worker
let (host_pid, worker) = worker_service.create_worker(
    &pod_id,
    "container1",
    &worker_info,
    gpu_observer,
).await?;

// Health checks
let is_healthy = worker_service.check_worker_health(&worker).await?;

// Graceful shutdown
worker_service.stop_worker(&pod_id, host_pid).await?;
```

#### Resource Monitor
```rust
use crate::domain::pod_management::services::{ResourceMonitor, PrometheusMetricsCollector};

let mut monitor = ResourceMonitor::new(device_service);

// Add Prometheus metrics
monitor.add_metrics_collector(Arc::new(PrometheusMetricsCollector::new()));

// Start monitoring
monitor.start_monitoring(registry, cancellation_token).await;

// Check violations
let violations = monitor.get_violations(&pod_id).await;
```

## Metrics and Monitoring

### Prometheus Integration

The module provides comprehensive Prometheus metrics:

```rust
use crate::domain::pod_management::services::{init_metrics, get_metrics};

// Initialize metrics registry
init_metrics()?;

// Export metrics
let metrics_text = get_metrics()?;
```

### Available Metrics

#### Pod Metrics
- `pod_management_pods_total` - Total number of managed pods
- `pod_management_workers_total` - Workers per pod
- `pod_management_healthy_workers` - Healthy workers per pod

#### Resource Metrics
- `pod_management_cpu_usage_percent` - CPU usage per pod
- `pod_management_memory_usage_bytes` - Memory usage per pod
- `pod_management_gpu_usage_percent` - GPU usage per device

#### Violation Metrics
- `pod_management_resource_violations_total` - Resource violations count
- `pod_management_violation_duration_seconds` - Violation duration

#### Operational Metrics
- `pod_management_monitoring_cycles_total` - Monitoring cycles completed
- `pod_management_monitoring_errors_total` - Monitoring errors

## Error Handling

### Error Types

```rust
use crate::domain::pod_management::core::{PodManagementError, Result};

match operation().await {
    Ok(result) => { /* success */ },
    Err(PodManagementError::PodNotFound(id)) => {
        eprintln!("Pod {} not found", id);
    },
    Err(PodManagementError::ResourceLimitExceeded(msg)) => {
        eprintln!("Resource limit exceeded: {}", msg);
    },
    Err(PodManagementError::DeviceAllocationFailed(msg)) => {
        eprintln!("Device allocation failed: {}", msg);
    },
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

### Error Context

All errors include rich context information:

```rust
// Errors automatically convert from underlying types
let result: Result<()> = nvml_operation().map_err(PodManagementError::from)?;

// Chain errors with context
operation().with_context(|| format!("Failed to process pod {}", pod_id))?;
```

## Testing

### Unit Tests

Comprehensive unit tests are provided for all components:

```bash
# Run all pod management tests
cargo test pod_management

# Run specific module tests
cargo test pod_management::core::registry
cargo test pod_management::services::device_service

# Run with detailed output
cargo test pod_management -- --nocapture
```

### Integration Tests

```rust
#[tokio::test]
async fn test_full_pod_lifecycle() {
    let registry = PodRegistry::new();
    let pod = create_test_pod();
    
    // Full lifecycle test
    registry.register_pod(pod).await.unwrap();
    // ... add workers, monitor resources, cleanup
}
```

### Mock Support

Test utilities are provided for mocking dependencies:

```rust
// Mock worker info for testing
fn create_test_worker_info() -> WorkerInfo {
    WorkerInfo {
        namespace: "test-namespace".to_string(),
        pod_name: "test-pod".to_string(),
        // ...
    }
}
```

## Performance Optimization

### Memory Management
- **Arc Usage**: Shared ownership where needed, unique ownership where possible
- **Async Efficiency**: Non-blocking operations throughout
- **Cache Efficiency**: Optimized data layout and access patterns

### Concurrency
- **Lock Granularity**: Fine-grained locking to minimize contention
- **Lock-free Operations**: Where possible, using atomic operations
- **Parallel Processing**: Concurrent monitoring and processing

### Resource Cleanup
- **Automatic Cleanup**: RAII and Drop traits ensure proper cleanup
- **Leak Detection**: Monitoring for resource leaks
- **Graceful Shutdown**: Proper cancellation token propagation

## Configuration

### Monitor Configuration
```rust
use crate::domain::pod_management::services::MonitorConfig;

let config = MonitorConfig {
    check_interval: Duration::from_secs(5),
    cpu_threshold: 90.0,
    memory_threshold: 85.0,
    gpu_threshold: 95.0,
    max_violations: 3,
};

let monitor = ResourceMonitor::with_config(device_service, config);
```

### Service Configuration
```rust
// Worker service with custom timeout
let mut worker_service = WorkerService::new(/* deps */);
worker_service.set_worker_timeout(Duration::from_secs(60));
```

## Migration Guide

### From Legacy Implementation

The new pod management module provides backward compatibility through deprecated re-exports:

```rust
// Old usage (deprecated)
use crate::domain::pod_management::LimiterCoordinator;

// New usage (recommended)
use crate::domain::pod_management::services::DeviceService;
```

### Migration Steps

1. **Update Imports**: Replace old imports with new service-based imports
2. **Update Error Handling**: Use the new unified error types
3. **Update Types**: Use strong-typed IDs instead of strings
4. **Add Metrics**: Integrate Prometheus metrics for monitoring
5. **Add Tests**: Use the new testing utilities

## Best Practices

### Type Safety
```rust
// ✅ Good: Use strong types
let pod_id = PodId::new(namespace, pod_name);

// ❌ Bad: Use raw strings
let pod_id = format!("tf_shm_{}_{}", namespace, pod_name);
```

### Error Handling
```rust
// ✅ Good: Use Result types consistently
async fn operation() -> Result<T> {
    // ... implementation
}

// ❌ Bad: Panic on errors
async fn operation() -> T {
    // ... implementation that might panic
}
```

### Resource Management
```rust
// ✅ Good: Use RAII and proper cleanup
{
    let resource = acquire_resource();
    // resource automatically cleaned up on drop
}

// ❌ Bad: Manual cleanup
let resource = acquire_resource();
// ... forget to cleanup
```

### Async Best Practices
```rust
// ✅ Good: Use cancellation tokens
async fn long_running_task(cancellation_token: CancellationToken) {
    tokio::select! {
        _ = cancellation_token.cancelled() => return,
        result = actual_work() => handle_result(result),
    }
}

// ❌ Bad: No cancellation support
async fn long_running_task() {
    loop {
        // ... no way to cancel
    }
}
```

## Troubleshooting

### Common Issues

#### Pod Not Found Errors
```rust
// Check pod exists before operations
if let Some(pod) = registry.get_pod(&pod_id).await {
    // proceed with operations
} else {
    return Err(PodManagementError::PodNotFound(pod_id.to_string()));
}
```

#### Resource Violations
```rust
// Monitor for violations and take action
let violations = monitor.get_violations(&pod_id).await;
for violation in violations {
    match violation.violation_type {
        ViolationType::GpuUsage { device_idx } => {
            // throttle GPU usage
        },
        ViolationType::MemoryUsage => {
            // trigger memory cleanup
        },
        _ => {}
    }
}
```

### Debugging

Enable detailed logging:
```rust
// Enable trace-level logging for pod management
RUST_LOG=hypervisor::domain::pod_management=trace cargo run
```

Check metrics for health:
```bash
# Check Prometheus metrics
curl http://localhost:8080/metrics | grep pod_management
```

## Contributing

### Adding New Features

1. **Types**: Add new types to `types/` module
2. **Services**: Add new services to `services/` module
3. **Tests**: Add comprehensive tests
4. **Documentation**: Update this README
5. **Metrics**: Add relevant Prometheus metrics

### Code Style

- Follow Rust naming conventions
- Use `Result<T>` for fallible operations
- Document all public APIs
- Add unit tests for all new functionality
- Use strong typing over primitives

## Examples

See the `tests/` directory for comprehensive examples of using the pod management module in various scenarios.