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

/// auto-executed function when module is loaded, used for test environment initialization
#[cfg(test)]
#[ctor::ctor]
fn init_test_environment() {
    init_test_logging();
}
