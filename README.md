# vgpu.rs

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs.svg?type=shield&issueType=license)](https://app.fossa.com/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs?ref=badge_shield&issueType=license)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs.svg?type=shield&issueType=security)](https://app.fossa.com/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs?ref=badge_shield&issueType=security)
[![Release](https://github.com/NexusGPU/vgpu.rs/actions/workflows/release.yml/badge.svg)](https://github.com/NexusGPU/vgpu.rs/actions/workflows/release.yml) [![Lint](https://github.com/NexusGPU/vgpu.rs/actions/workflows/lint.yml/badge.svg)](https://github.com/NexusGPU/vgpu.rs/actions/workflows/lint.yml) [![Test](https://github.com/NexusGPU/vgpu.rs/actions/workflows/test.yml/badge.svg)](https://github.com/NexusGPU/vgpu.rs/actions/workflows/test.yml)

vgpu.rs is the fractional GPU & vgpu-hypervisor implementation written in Rust

## Project Structure

This project is organized as a Cargo workspace containing multiple crates, each with specific responsibilities:

### Crates

- [**hypervisor**](crates/hypervisor): The hypervisor implementation that monitors and manages GPU resources. It leverages NVML (NVIDIA Management Library) to track GPU utilization and optimize CUDA workload scheduling.

- [**cuda-limiter**](crates/cuda-limiter): A dynamic library that intercepts CUDA API calls to enforce resource limits and scheduling policies. Built as a `cdylib` that can be preloaded into CUDA applications to control their resource usage.

- [**add-path**](crates/add-path): A utility library that modifies environment variables like `PATH`, `LD_PRELOAD`, and `LD_LIBRARY_PATH` to ensure proper library loading and execution. Built as a `cdylib` for runtime loading.
  
  This library supports both appending and prepending values to environment variables:
  - By default, when an environment variable such as `TF_PATH`, `TF_LD_PRELOAD`, or `TF_LD_LIBRARY_PATH` is set, its value will be appended to the corresponding variable (e.g., `PATH`).
  - If you want to prepend a value instead (i.e., place it at the beginning), use an environment variable prefixed with `TF_PREPEND_`, such as `TF_PREPEND_PATH`. This will insert the value at the front, ensuring it takes precedence during library or binary lookup.
  
  This flexible mechanism allows fine-grained control over environment variable ordering, which is critical for correct library loading and runtime behavior in complex CUDA or GPU environments.

- [**macro**](crates/macro): Contains procedural macros that simplify common patterns used throughout the codebase, improving code readability and reducing boilerplate.

- [**utils**](crates/utils): A collection of common utilities and helper functions shared across the project. Includes tracing, logging, and other infrastructure components.
