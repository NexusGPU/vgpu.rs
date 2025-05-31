use std::sync::LazyLock;

use crate::build_info::BUILD_INFO;

/// Defines the application version.
pub static VERSION: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}-{}{}",
        env!("IMAGE_VERSION"),
        BUILD_INFO.commit_sha1.unwrap_or("unknown"),
        if BUILD_INFO.git_dirty { "-dirty" } else { "" }
    )
});
