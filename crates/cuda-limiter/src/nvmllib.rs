use std::env;
use std::ffi::{OsStr, OsString};

use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;

const PRIMARY_NVML_LIB: &str = "libnvidia-ml.so.1";
const NVML_FALLBACK_LIB: &str = "libnvidia-ml.so";

pub(crate) fn init_nvml() -> Result<Nvml, NvmlError> {
    let mut last_err: Option<NvmlError> = None;
    let mut candidates: Vec<OsString> = Vec::with_capacity(4);

    if let Some(path) = env::var_os("TF_NVML_LIB_PATH") {
        candidates.push(path);
    }

    candidates.push(OsStr::new(PRIMARY_NVML_LIB).to_os_string());
    candidates.push(OsStr::new(NVML_FALLBACK_LIB).to_os_string());

    for candidate in candidates {
        let candidate_display = candidate.to_string_lossy();
        tracing::info!("Loading NVML library from {}", candidate_display);
        match Nvml::builder().lib_path(candidate.as_os_str()).init() {
            Ok(nvml) => return Ok(nvml),
            Err(err) => {
                tracing::warn!(error = %err, "Failed to load {}", candidate_display);
                last_err = Some(err);
            }
        }
    }

    Err(last_err.unwrap_or(NvmlError::Unknown))
}
