use std::ffi;

use nvml_wrapper::Nvml;
use once_cell::sync::OnceCell;
use tracing::info;

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
}
