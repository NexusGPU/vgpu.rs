# vgpu.rs

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs.svg?type=shield&issueType=license)](https://app.fossa.com/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs?ref=badge_shield&issueType=license)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs.svg?type=shield&issueType=security)](https://app.fossa.com/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs?ref=badge_shield&issueType=security)
[![Release](https://github.com/NexusGPU/vgpu.rs/actions/workflows/release.yml/badge.svg)](https://github.com/NexusGPU/vgpu.rs/actions/workflows/release.yml)
[![Lint](https://github.com/NexusGPU/vgpu.rs/actions/workflows/lint.yml/badge.svg)](https://github.com/NexusGPU/vgpu.rs/actions/workflows/lint.yml)
[![Test](https://github.com/NexusGPU/vgpu.rs/actions/workflows/test.yml/badge.svg)](https://github.com/NexusGPU/vgpu.rs/actions/workflows/test.yml)

vgpu.rs is the fractional GPU & vgpu-hypervisor implementation written in Rust

## Installation

You can download the latest release binaries from the
[GitHub Releases page](https://github.com/NexusGPU/vgpu.rs/releases). Use the
following command to automatically download the appropriate version

```bash
# Download and extract the latest release
ARCH=$(uname -m | sed 's/x86_64/x64/' | sed 's/aarch64/arm64/')

# Download the libraries
wget "https://github.com/NexusGPU/vgpu.rs/releases/latest/download/libadd_path-${ARCH}.tar.gz"
wget "https://github.com/NexusGPU/vgpu.rs/releases/latest/download/libcuda_limiter-${ARCH}.tar.gz"

# Extract the archives
tar -xzf libadd_path-${ARCH}.tar.gz
tar -xzf libcuda_limiter-${ARCH}.tar.gz

# Optional: Remove the archives after extraction
rm libadd_path-${ARCH}.tar.gz libcuda_limiter-${ARCH}.tar.gz
```

## Usage

### Using cuda-limiter

The `cuda-limiter` library intercepts CUDA API calls to enforce resource limits. After downloading and extracting the library, you can
use it as follows:

```bash

# First, get your GPU UUIDs or device indices
# Run this command to list all available GPUs and their UUIDs
nvidia-smi -L
# Example output:
GPU 0: NVIDIA GeForce RTX 4060 Ti (UUID: GPU-3430f778-7a25-704c-9090-8b0bb2478114)

# Set environment variables to configure limits
# You can use either GPU UUIDs (case-insensitive), device indices, or both as keys
# 1. Use only GPU UUID as key
export TENSOR_FUSION_CUDA_UP_LIMIT='{"gpu-3430f778-7a25-704c-9090-8b0bb2478114": 10}'
export TENSOR_FUSION_CUDA_MEM_LIMIT='{"gpu-3430f778-7a25-704c-9090-8b0bb2478114": 1073741824}'

# 2. Use only device index as key
export TENSOR_FUSION_CUDA_UP_LIMIT='{"0": 20}'
export TENSOR_FUSION_CUDA_MEM_LIMIT='{"0": 2147483648}'

# Preload the cuda-limiter library and run an application
LD_PRELOAD=/path/to/libcuda_limiter.so your_cuda_application

# To verify the limiter is working, check nvidia-smi output
LD_PRELOAD=/path/to/libcuda_limiter.so nvidia-smi
# Nvidia-smi output:
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 Ti     Off |   00000000:01:00.0 Off |                  N/A |
|  0%   37C    P8             11W /  160W |       0MiB /   1024MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|                                                                                         |
+-----------------------------------------------------------------------------------------+


```

### Using add-path

The `add-path` library modifies environment variables at runtime to ensure
proper library loading and execution. After downloading and extracting the
library, you need to load it using LD_PRELOAD:

```bash
# First, set LD_PRELOAD to use the add-path library
export LD_PRELOAD=/path/to/libadd_path.so

# Basic usage: Add a path to the PATH environment variable
sh -c "TF_PATH=/usr/custom/bin env | grep PATH"
# Example output: PATH=/home/user/bin:/usr/local/sbin:/bin:/usr/custom/bin

# Add a path to LD_PRELOAD
sh -c "TF_LD_PRELOAD=/path/to/custom.so env | grep LD_PRELOAD"
# Example output: LD_PRELOAD=/path/to/libadd_path.so:/path/to/custom.so

# Add a path to LD_LIBRARY_PATH
sh -c "TF_LD_LIBRARY_PATH=/usr/local/cuda/lib64 env | grep LD_LIBRARY_PATH"
# Example output: LD_LIBRARY_PATH=/usr/lib64:/lib64:/usr/local/cuda/lib64

# Prepend a path to PATH (higher priority)
sh -c "TF_PREPEND_PATH=/opt/custom/bin env | grep PATH"
# Example output: PATH=/opt/custom/bin:/home/user/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

**Note**: The key difference between using `TF_PATH` and `TF_PREPEND_PATH` is the position where the path is added. Prepending puts the path at the beginning, giving it higher priority when the system searches for executables or libraries.

## Project Structure

This project is organized as a Cargo workspace containing multiple crates, each
with specific responsibilities:

### Crates

- [**hypervisor**](crates/hypervisor): The hypervisor implementation that
  monitors and manages GPU resources. It leverages NVML (NVIDIA Management
  Library) to track GPU utilization and optimize CUDA workload scheduling.

- [**cuda-limiter**](crates/cuda-limiter): A dynamic library that intercepts
  CUDA API calls to enforce resource limits. Built as a
  `cdylib` that can be preloaded into CUDA applications to control their
  resource usage.
  
  > **Implementation Reference**: The cuda-limiter module's design and implementation is based on research from
  > [GaiaGPU: Sharing GPUs in Container Clouds](https://ieeexplore.ieee.org/document/8672318). This paper introduces
  > innovative techniques for GPU resource management and isolation in container environments.

- [**add-path**](crates/add-path): A utility library that modifies environment
  variables like `PATH`, `LD_PRELOAD`, and `LD_LIBRARY_PATH` to ensure proper
  library loading and execution. Built as a `cdylib` for runtime loading.

  This library supports both appending and prepending values to environment
  variables:
  - By default, when an environment variable such as `TF_PATH`, `TF_LD_PRELOAD`,
    or `TF_LD_LIBRARY_PATH` is set, its value will be appended to the
    corresponding variable (e.g., `PATH`).
  - If you want to prepend a value instead (i.e., place it at the beginning),
    use an environment variable prefixed with `TF_PREPEND_`, such as
    `TF_PREPEND_PATH`. This will insert the value at the front, ensuring it
    takes precedence during library or binary lookup.

  This flexible mechanism allows fine-grained control over environment variable
  ordering, which is critical for correct library loading and runtime behavior
  in complex CUDA or GPU environments.

- [**macro**](crates/macro): Contains procedural macros that simplify common
  patterns used throughout the codebase, improving code readability and reducing
  boilerplate.

- [**utils**](crates/utils): A collection of common utilities and helper
  functions shared across the project. Includes tracing, logging, and other
  infrastructure components.
