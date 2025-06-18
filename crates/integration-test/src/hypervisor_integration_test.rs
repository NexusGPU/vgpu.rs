use std::collections::HashMap;
use std::fs;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use nvml_wrapper::Nvml;
use tracing::info;

use crate::integration_framework::is_cuda_available;
use crate::integration_framework::ClientConfig;
use crate::integration_framework::IntegrationTestSetup;
use crate::integration_framework::MemoryPattern;
use crate::integration_framework::QosLevel;

const TFLOP_LIMIT: u32 = 10;

fn get_total_gpu_memory(gpu_index: u32) -> Result<u64> {
    let nvml = Nvml::init()?;
    let device = nvml.device_by_index(gpu_index)?;
    let memory_info = device.memory_info()?;
    Ok(memory_info.total)
}

#[test]
fn test_priority_scheduling_with_metrics() {
    if !is_cuda_available() {
        info!("CUDA not available, skipping test");
        return;
    }

    let mut setup = IntegrationTestSetup::new().expect("Failed to create test setup");

    // Enable metrics logging
    let metrics_file = setup
        .enable_metrics_logging()
        .expect("Failed to enable metrics logging");

    // Start hypervisor
    setup
        .start_hypervisor()
        .expect("Failed to start hypervisor");

    // Get total GPU memory and define allocation size that forces competition
    let total_mem = get_total_gpu_memory(0).expect("Failed to get GPU memory");
    let alloc_size = (total_mem as f64 * 0.7) as u64; // Allocate 70% of memory

    // Start workers with different QoS levels
    let _high_worker = setup
        .start_worker(QosLevel::High, total_mem, TFLOP_LIMIT)
        .expect("Failed to start high QoS worker");
    let _medium_worker = setup
        .start_worker(QosLevel::Medium, total_mem, TFLOP_LIMIT)
        .expect("Failed to start medium QoS worker");
    let _low_worker = setup
        .start_worker(QosLevel::Low, total_mem, TFLOP_LIMIT)
        .expect("Failed to start low QoS worker");

    // Give workers time to register
    std::thread::sleep(Duration::from_secs(3));

    let client_duration = Duration::from_secs(10);

    // Client configs
    let high_client_config = ClientConfig {
        memory_pattern: MemoryPattern::SingleLargeAlloc(alloc_size),
        duration: client_duration,
        gpu_index: 0,
    };
    let medium_client_config = ClientConfig {
        memory_pattern: MemoryPattern::SingleLargeAlloc(alloc_size),
        duration: client_duration,
        gpu_index: 0,
    };
    let low_client_config = ClientConfig {
        memory_pattern: MemoryPattern::SingleLargeAlloc(alloc_size),
        duration: client_duration,
        gpu_index: 0,
    };

    // Start clients in reverse priority to test preemption
    info!("Starting clients in reverse priority order...");
    let low_client_pid = setup
        .start_client(low_client_config)
        .expect("Failed to start low priority client");
    std::thread::sleep(Duration::from_secs(1)); // Stagger starts slightly

    let medium_client_pid = setup
        .start_client(medium_client_config)
        .expect("Failed to start medium priority client");
    std::thread::sleep(Duration::from_secs(1));

    let high_client_pid = setup
        .start_client(high_client_config)
        .expect("Failed to start high priority client");
    info!("All clients started. Waiting for completion...");

    let mut completion_times = HashMap::new();
    let client_pids = [
        ("high", high_client_pid),
        ("medium", medium_client_pid),
        ("low", low_client_pid),
    ];

    for (name, pid) in client_pids {
        match setup.wait_for_client(pid, Duration::from_secs(100)) {
            Ok(output) => {
                completion_times.insert(name, Instant::now());
                info!(
                    "Client {} (pid: {}) completed with status: {}",
                    name, pid, output.status
                );
            }
            Err(e) => {
                panic!("Client {name} (pid: {pid}) failed to complete: {e}");
            }
        }
    }

    // --- Assertions ---

    // 1. Assert completion order
    assert!(
        completion_times.get("high") < completion_times.get("medium"),
        "High priority client should finish before medium"
    );
    assert!(
        completion_times.get("medium") < completion_times.get("low"),
        "Medium priority client should finish before low"
    );
    info!("Client completion order is correct (High -> Medium -> Low).");

    // 2. Assert scheduling decisions from metrics file
    let metrics_content = fs::read_to_string(&metrics_file)
        .unwrap_or_else(|e| panic!("Failed to read metrics file at {metrics_file:?}: {e}"));

    info!("--- Metrics Log ---");
    info!("{}", metrics_content);
    info!("--- End Metrics Log ---");

    // Verify that low and medium clients were released for the high priority one
    assert!(
        metrics_content.contains(&format!(
            "decision_type=\"release\",pid=\"{low_client_pid}\""
        )),
        "Metrics should show release decision for low priority client"
    );
    assert!(
        metrics_content.contains(&format!(
            "decision_type=\"release\",pid=\"{medium_client_pid}\""
        )),
        "Metrics should show release decision for medium priority client"
    );

    // Verify that clients were resumed in the correct order
    let high_resume_pos = metrics_content
        .find(&format!(
            "decision_type=\"resume\",pid=\"{high_client_pid}\""
        ))
        .unwrap_or(usize::MAX);
    let medium_resume_pos = metrics_content
        .find(&format!(
            "decision_type=\"resume\",pid=\"{medium_client_pid}\""
        ))
        .unwrap_or(usize::MAX);
    let low_resume_pos = metrics_content
        .find(&format!(
            "decision_type=\"resume\",pid=\"{low_client_pid}\""
        ))
        .unwrap_or(usize::MAX);

    // The first one to start might not have a "Resume" if it gets to run immediately.
    // However, since we start low -> medium -> high, low and medium should be released,
    // and then all should be resumed in order of priority.
    // The logic is: high should be woken up first, then medium resumed, then low resumed.
    // "Wake" is for a trapped process, "Resume" is for a sleeping (released) process.
    // The test logic might cause a mix of Wake and Resume. Let's check relative order.
    let high_wake_pos = metrics_content
        .find(&format!("decision_type=\"wake\",pid=\"{high_client_pid}\""))
        .unwrap_or(usize::MAX);

    let high_pos = high_resume_pos.min(high_wake_pos);

    assert!(
        high_pos < medium_resume_pos,
        "High priority client should be resumed/woken before medium"
    );
    assert!(
        medium_resume_pos < low_resume_pos,
        "Medium priority client should be resumed before low"
    );

    info!("Scheduling decisions in metrics log are correct.");

    assert!(
        setup.is_hypervisor_running(),
        "Hypervisor should still be running"
    );
}

#[test]
fn test_worker_disconnection() {
    if !is_cuda_available() {
        info!("CUDA not available, skipping test");
        return;
    }

    let mut setup = IntegrationTestSetup::new().expect("Failed to create test setup");

    // Start hypervisor
    setup
        .start_hypervisor()
        .expect("Failed to start hypervisor");

    // Start worker
    let _worker_pid = setup
        .start_worker(QosLevel::Medium, 256 * 1024 * 1024, TFLOP_LIMIT)
        .expect("Failed to start worker");

    std::thread::sleep(Duration::from_secs(3));

    assert_eq!(
        setup.running_worker_count(),
        1,
        "Should have 1 worker running"
    );

    // Kill the worker process
    setup.stop_worker(0).expect("Failed to stop worker");

    // Give hypervisor time to detect the disconnection
    std::thread::sleep(Duration::from_secs(5));

    // Verify hypervisor is still running and handled the disconnection gracefully
    assert!(
        setup.is_hypervisor_running(),
        "Hypervisor should still be running after worker disconnection"
    );

    info!("Worker disconnection test completed successfully");
}
