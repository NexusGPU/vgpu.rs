use std::fs;
use std::os::unix::fs::symlink;
use std::process;

// Constants
const TENSOR_FUSION_DIR: &str = "/tensor-fusion";
const SOURCE_DIR: &str = "/home/app";
const NVIDIA_SMI_SOURCE: &str = "nvidia-smi-linux";
const PRELOAD_CONFIG_PATH: &str = "/tensor-fusion/ld.so.preload";
const PRELOAD_ADD_PATH_LIBS: &str = "/tensor-fusion/libadd_path.so\n";
const PRELOAD_CUDA_STUB_LIBS: &str = "/tensor-fusion/libcuda.so\n";
const PRELOAD_NGPU_LIBS: &str = "/tensor-fusion/libcuda_limiter.so\n";
const NVIDIA_ML_LIB: &str = "/tensor-fusion/libnvidia-ml.so";
const NVIDIA_ML_SYMLINK: &str = "/tensor-fusion/libnvidia-ml.so.1";
const NVIDIA_SMI_TARGET: &str = "/tensor-fusion/nvidia-smi";

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
    log_info("Starting tensor-fusion setup");

    if let Err(e) = run_setup() {
        log_error(&format!("Setup failed: {e}"));
        process::exit(1);
    }

    log_info("Setup completed successfully");
}

fn run_setup() -> Result<()> {
    write_ld_preload_config()?;
    copy_dynamic_lib_files()?;
    copy_nvidia_smi_to_path()?;
    create_symlink_for_nvml()?;
    Ok(())
}

fn write_ld_preload_config() -> Result<()> {
    log_info("Writing ld.so.preload config");
    fs::write(
        PRELOAD_CONFIG_PATH,
        if is_ngpu_mode() {
            log_info("nGPU mode detected");
            if std::env::var("DISABLE_GPU_LIMITER").unwrap_or_default() == "true" {
                log_info("GPU limiter disabled");
                format!("{PRELOAD_ADD_PATH_LIBS}{PRELOAD_CUDA_STUB_LIBS}")
            } else {
                format!("{PRELOAD_ADD_PATH_LIBS}{PRELOAD_NGPU_LIBS}")
            }
        } else {
            log_info("vGPU mode detected");
            format!("{PRELOAD_ADD_PATH_LIBS}{PRELOAD_CUDA_STUB_LIBS}")
        },
    )
    .map_err(|e| format!("Failed to write preload config to {PRELOAD_CONFIG_PATH}: {e}",))?;
    log_debug(&format!(
        "Successfully wrote preload config to {PRELOAD_CONFIG_PATH}",
    ));
    Ok(())
}

fn copy_dynamic_lib_files() -> Result<()> {
    log_info(&format!("Copying dynamic lib files from {SOURCE_DIR}"));

    let entries = fs::read_dir(SOURCE_DIR)
        .map_err(|e| format!("Failed to read directory {SOURCE_DIR}: {e}"))?;

    let mut copied_count = 0;
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "so") {
            let filename = path
                .file_name()
                .ok_or("Invalid filename")?
                .to_string_lossy();

            let target_path = format!("{TENSOR_FUSION_DIR}/{filename}");
            log_debug(&format!("Copying {} to {}", path.display(), target_path));

            fs::copy(&path, &target_path).map_err(|e| {
                format!("Failed to copy {} to {}: {e}", path.display(), target_path,)
            })?;

            copied_count += 1;
        }
    }

    log_info(&format!(
        "Successfully copied {copied_count} dynamic lib files",
    ));
    Ok(())
}

fn copy_nvidia_smi_to_path() -> Result<()> {
    log_info("Copying nvidia-smi binary");
    fs::copy(NVIDIA_SMI_SOURCE, NVIDIA_SMI_TARGET)
        .map_err(|e| format!("Failed to copy {NVIDIA_SMI_SOURCE} to {NVIDIA_SMI_TARGET}: {e}",))?;
    log_debug(&format!(
        "Successfully copied {NVIDIA_SMI_SOURCE} to {NVIDIA_SMI_TARGET}",
    ));
    Ok(())
}

fn create_symlink_for_nvml() -> Result<()> {
    log_info("Creating nvidia-ml symlink");

    // Remove existing symlink if it exists
    if fs::metadata(NVIDIA_ML_SYMLINK).is_ok() {
        fs::remove_file(NVIDIA_ML_SYMLINK)
            .map_err(|e| format!("Failed to remove existing symlink {NVIDIA_ML_SYMLINK}: {e}",))?;
        log_debug(&format!("Removed existing symlink {NVIDIA_ML_SYMLINK}"));
    }

    symlink(NVIDIA_ML_LIB, NVIDIA_ML_SYMLINK).map_err(|e| {
        format!("Failed to create symlink from {NVIDIA_ML_LIB} to {NVIDIA_ML_SYMLINK}: {e}",)
    })?;
    log_debug(&format!(
        "Successfully created symlink from {NVIDIA_ML_LIB} to {NVIDIA_ML_SYMLINK}",
    ));
    Ok(())
}

fn is_ngpu_mode() -> bool {
    fs::metadata(format!("{SOURCE_DIR}/libcuda_limiter.so")).is_ok()
}
// Logging utilities
fn log_info(msg: &str) {
    println!("[INFO] {msg}");
}

fn log_debug(msg: &str) {
    println!("[DEBUG] {msg}");
}

fn log_error(msg: &str) {
    eprintln!("[ERROR] {msg}");
}
