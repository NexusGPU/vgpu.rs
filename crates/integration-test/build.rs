use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use serde::Deserialize;

fn main() {
    // get output directory
    let out_dir = env::var("OUT_DIR").unwrap();
    let target_dir = PathBuf::from(&out_dir);

    // Download tensor-fusion components
    download_tensor_fusion_components(&target_dir);

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

    // Provide paths to tensor-fusion components
    println!(
        "cargo:rustc-env=TENSOR_FUSION_WORKER_PATH={}",
        target_dir.join("tensor-fusion-worker").display()
    );
    println!(
        "cargo:rustc-env=TENSOR_FUSION_LIBCUDA_PATH={}",
        target_dir.join("libcuda.so").display()
    );
    println!(
        "cargo:rustc-env=TENSOR_FUSION_LIBTELEPORT_PATH={}",
        target_dir.join("libteleport.so").display()
    );
    println!(
        "cargo:rustc-env=TENSOR_FUSION_LIBNVML_PATH={}",
        target_dir.join("libnvidia-ml.so").display()
    );
}

fn download_tensor_fusion_components(target_dir: &PathBuf) {
    println!("cargo:warning=Downloading tensor-fusion components...");

    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .build()
        .unwrap();

    let latest_version = get_latest_release_version(&client).unwrap();

    let archive_filename = "tensor-fusion-components.tar.gz";
    let archive_path = target_dir.join(archive_filename);
    let download_url = format!(
        "https://github.com/NexusGPU/vgpu.rs/releases/download/{latest_version}/{archive_filename}",
    );

    if archive_path.exists() {
        println!(
            "cargo:warning=Archive {archive_filename} already exists, skipping download",
        );
    } else {
        println!(
            "cargo:warning=Downloading {archive_filename} from {download_url}",
        );

        match download_file(&client, &download_url, &archive_path) {
            Ok(_) => {
                println!("cargo:warning=Successfully downloaded {archive_filename}");
            }
            Err(e) => {
                panic!("Failed to download {archive_filename}: {e}");
            }
        }
    }

    // extract the archive
    println!("cargo:warning=Extracting {archive_filename}");
    match extract_tar_gz(&archive_path, target_dir) {
        Ok(_) => {
            println!(
                "cargo:warning=Successfully extracted tensor-fusion components to {}",
                target_dir.display()
            );

            // set executable permissions for the worker binary
            let worker_path = target_dir.join("tensor-fusion-worker");
            set_executable_permissions(&worker_path);

            // remove the archive to save space
            if let Err(e) = fs::remove_file(&archive_path) {
                println!("cargo:warning=Failed to remove archive file: {e}");
            }
        }
        Err(e) => {
            // If extraction fails, it's likely the archive is corrupt.
            // Remove it so the next build will re-download it.
            let _ = fs::remove_file(&archive_path);
            panic!(
                "Failed to extract {archive_filename}: {e}. The archive has been removed. Please try building again.",
            );
        }
    }

    println!("cargo:warning=Tensor-fusion components download completed");
}

fn download_file(
    client: &reqwest::blocking::Client,
    url: &str,
    file_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = client.get(url).send()?;

    if !response.status().is_success() {
        return Err(format!("HTTP error: {}", response.status()).into());
    }

    let mut file = fs::File::create(file_path)?;
    let content = response.bytes()?;
    file.write_all(&content)?;

    Ok(())
}

fn set_executable_permissions(file_path: &PathBuf) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = fs::metadata(file_path) {
            let mut perms = metadata.permissions();
            perms.set_mode(0o755);
            if let Err(e) = fs::set_permissions(file_path, perms) {
                println!(
                    "cargo:warning=Failed to set executable permissions for {file_path:?}: {e}",
                );
            }
        }
    }
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
}

fn get_latest_release_version(
    client: &reqwest::blocking::Client,
) -> Result<String, Box<dyn std::error::Error>> {
    let api_url = "https://api.github.com/repos/NexusGPU/vgpu.rs/releases/latest";

    let response = client
        .get(api_url)
        .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36")
        .send()?;

    if !response.status().is_success() {
        return Err(format!("HTTP error: {}", response.status()).into());
    }

    let release: Release = response.json()?;
    Ok(release.tag_name)
}

fn extract_tar_gz(
    archive_path: &PathBuf,
    extract_to: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = fs::File::open(archive_path)?;
    let tar = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(tar);

    // extract the files
    archive.unpack(extract_to)?;

    Ok(())
}
