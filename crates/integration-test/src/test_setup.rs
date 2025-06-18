use std::ffi;
use std::process::Command;

use nvml_wrapper::Nvml;
use once_cell::sync::OnceCell;
use tracing::info;
use tracing::warn;

/// global logging initialization function, ensure logging is initialized only once
pub fn init_test_logging() {
    static INIT: OnceCell<()> = OnceCell::new();
    INIT.get_or_init(|| {
        utils::logging::init();
        info!("Test logging initialized");
    });
}

pub fn global_nvml() -> &'static Nvml {
    static INIT: OnceCell<Nvml> = OnceCell::new();
    INIT.get_or_init(|| {
        let nvml = match Nvml::init() {
            Ok(nvml) => nvml,
            Err(_) => Nvml::builder()
                .lib_path(ffi::OsStr::new("libnvidia-ml.so.1"))
                .init()
                .expect("Failed to initialize NVML"),
        };
        nvml
    })
}

/// auto-executed function when module is loaded, used for test environment initialization
#[cfg(test)]
#[ctor::ctor]
fn init_test_environment() {
    init_test_logging();
    build_dependent_crates();
}

fn build_dependent_crates() {
    info!("Building dependent crates for integration tests...");
    let crates = ["cuda-limiter", "hypervisor"];
    for &crate_name in &crates {
        info!("Building {}...", crate_name);
        let output = Command::new("cargo")
            .args(["build", "--package", crate_name])
            .output()
            .unwrap_or_else(|e| panic!("Failed to execute cargo build for {crate_name}: {e}"));

        if !output.status.success() {
            warn!(
                "Failed to build {}:\nstdout: {}\nstderr: {}",
                crate_name,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        } else {
            info!("Successfully built {crate_name}.");
        }
    }
}
