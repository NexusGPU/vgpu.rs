use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn main() {
    // get output directory
    let out_dir = env::var("OUT_DIR").unwrap();
    let target_dir = PathBuf::from(&out_dir);

    // CUDA source file path
    let cuda_src_path = "src/cuda_test.cu";

    // rerun if CUDA source file changes
    println!("cargo:rerun-if-changed={cuda_src_path}");

    // Check if nvcc is available; if not, skip building the CUDA test program
    let nvcc_available = Command::new("nvcc")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !nvcc_available {
        println!("cargo:warning=nvcc not found; skipping CUDA test program build");
        // Provide a sentinel so code can detect absence and skip tests gracefully
        println!("cargo:rustc-env=CUDA_TEST_PROGRAM_PATH=__NVCC_NOT_FOUND__");
        return;
    }

    // compile CUDA code
    let nvcc_status = Command::new("nvcc")
        .arg(cuda_src_path)
        .arg("-o")
        .arg(target_dir.join("cuda_test_program"))
        .arg("-O3")
        .status()
        .expect("Failed to execute nvcc");

    if !nvcc_status.success() {
        panic!("Failed to compile CUDA code");
    }

    // notify Cargo to provide the path to the compiled CUDA program to the Rust program
    println!(
        "cargo:rustc-env=CUDA_TEST_PROGRAM_PATH={}",
        target_dir.join("cuda_test_program").display()
    );
}
