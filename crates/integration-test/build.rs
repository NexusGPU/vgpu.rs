use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // get output directory
    let out_dir = env::var("OUT_DIR").unwrap();
    let target_dir = PathBuf::from(&out_dir);

    // CUDA source file path
    let cuda_src_path = "src/cuda_test.cu";

    // rerun if CUDA source file changes
    println!("cargo:rerun-if-changed={cuda_src_path}");
    // compile CUDA code
    let nvcc_status = Command::new("nvcc")
        .arg(cuda_src_path)
        .arg("-o")
        .arg(target_dir.join("cuda_test_program"))
        // add appropriate CUDA architecture flags, e.g. compute_60 and sm_60 for Pascal architecture
        .arg("--gpu-architecture=sm_60")
        // optional: add optimization flags
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
