# Tensor Fusion Hypervisor

[![Release](https://github.com/NexusGPU/tensor-fusion-hypervisor/actions/workflows/release.yml/badge.svg)](https://github.com/NexusGPU/tensor-fusion-hypervisor/actions/workflows/release.yml) [![Lint](https://github.com/NexusGPU/tensor-fusion-hypervisor/actions/workflows/lint.yml/badge.svg)](https://github.com/NexusGPU/tensor-fusion-hypervisor/actions/workflows/lint.yml) [![Test](https://github.com/NexusGPU/tensor-fusion-hypervisor/actions/workflows/test.yml/badge.svg)](https://github.com/NexusGPU/tensor-fusion-hypervisor/actions/workflows/test.yml)

Tensor-Fusion-Hypervisor is a hypervisor that manages tensor-fusion worker scheduling, ensuring efficient GPU utilization and optimizing CUDA program performance.

## Project Structure

This project is organized as a Cargo workspace containing multiple crates, each with specific responsibilities:

### Crates

- **hypervisor** (`crates/hypervisor`): The main tensor fusion hypervisor implementation that monitors and manages GPU resources. It leverages NVML (NVIDIA Management Library) to track GPU utilization and optimize CUDA workload scheduling.

- **cuda-limiter** (`crates/cuda-limiter`): A dynamic library that intercepts CUDA API calls to enforce resource limits and scheduling policies. Built as a `cdylib` that can be preloaded into CUDA applications to control their resource usage.

- **add-path** (`crates/add-path`): A utility library that modifies environment variables like `PATH`, `LD_PRELOAD`, and `LD_LIBRARY_PATH` to ensure proper library loading and execution. Built as a `cdylib` for runtime loading.

- **macro** (`crates/macro`): Contains procedural macros (named `tf-macro`) that simplify common patterns used throughout the codebase, improving code readability and reducing boilerplate.

- **utils** (`crates/utils`): A collection of common utilities and helper functions shared across the project. Includes tracing, logging, and other infrastructure components.
