# vgpu.rs

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs.svg?type=shield&issueType=license)](https://app.fossa.com/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs?ref=badge_shield&issueType=license)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs.svg?type=shield&issueType=security)](https://app.fossa.com/projects/git%2Bgithub.com%2FNexusGPU%2Fvgpu.rs?ref=badge_shield&issueType=security)
[![Release](https://github.com/NexusGPU/vgpu.rs/actions/workflows/release.yml/badge.svg)](https://github.com/NexusGPU/vgpu.rs/actions/workflows/release.yml)
[![Lint](https://github.com/NexusGPU/vgpu.rs/actions/workflows/lint.yml/badge.svg)](https://github.com/NexusGPU/vgpu.rs/actions/workflows/lint.yml)
[![Test](https://github.com/NexusGPU/vgpu.rs/actions/workflows/test.yml/badge.svg)](https://github.com/NexusGPU/vgpu.rs/actions/workflows/test.yml)

vgpu.rs is a fractional GPU & vgpu-hypervisor implementation written in Rust.

## Installation

You can download the latest release binaries from the
[GitHub Releases page](https://github.com/NexusGPU/vgpu.rs/releases). Use the
following command to automatically download the appropriate version:

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

## System Architecture

### 1. Overview

This project provides a Kubernetes-based solution for GPU virtualization and resource limiting. Its primary goal is to enable multiple pods to share physical GPU resources in a multi-tenant environment while precisely controlling the GPU usage of each pod.

The system consists of two core components:

*   **Hypervisor (Daemon)**: A daemon process running on each GPU node, responsible for managing and scheduling all pods that require GPU resources on that node.
*   **Cuda-Limiter (SO Library)**: A dynamic library injected into the user application's process space via the `LD_PRELOAD` mechanism. It intercepts CUDA API calls, communicates with the Hypervisor, and enforces GPU operation limits based on the Hypervisor's scheduling decisions.

The entire system is designed to run in a Kubernetes cluster, using Pods as the basic unit for resource isolation and limitation. This means all processes (workers/limiters) within the same pod share the same GPU quota.

### 2. Core Components

#### 2.1. Hypervisor

The Hypervisor is the "brain" of the system, deployed as a DaemonSet on each Kubernetes node equipped with a GPU. Its main responsibilities include:

*   **Worker Management**: Tracks and manages all connected `cuda-limiter` instances (i.e., workers).
*   **Resource Monitoring**:
    *   Monitors physical GPU metrics (e.g., utilization, memory usage) via NVML (NVIDIA Management Library).
    *   Aggregates GPU usage data reported by all workers.
*   **Scheduling Policy**:
    *   Dynamically decides which pod's process is allowed to perform GPU computations based on a predefined scheduling algorithm (e.g., weighted round-robin) and each pod's resource quota.
    *   Scheduling decisions are dispatched to the corresponding `cuda-limiter` instances.
*   **Shared Memory Communication**:
    *   Creates a shared memory region to efficiently broadcast global GPU utilization data to all `cuda-limiter` instances on the node. This avoids the overhead of each worker querying NVML individually.
*   **Kubernetes Integration**:
    *   Interacts with the Kubernetes API Server to watch for pod creation and deletion events.
    *   Retrieves pod metadata (like pod name, resource requests) and associates this information with workers to achieve pod-level resource isolation.
*   **API Service**:
    *   Provides an HTTP API for `cuda-limiter` registration, command polling, status reporting, etc.
    *   Uses the `http-bidir-comm` crate to implement bidirectional communication with `cuda-limiter`.

#### 2.2. Cuda-Limiter

Cuda-Limiter is a dynamic library (`.so` file) injected into every user process that needs to use the GPU. Its core function is to act as the Hypervisor's agent within the user process.

*   **API Interception (Hooking)**:
    *   Uses techniques like `frida-gum` to intercept critical CUDA API calls (e.g., `cuLaunchKernel`) and NVML API calls at runtime.
    *   This is the key to enforcing GPU limits, as all GPU-related operations must first be inspected by `cuda-limiter`.
*   **Communication with Hypervisor**:
    *   On initialization, it obtains the Hypervisor's address from environment variables and registers itself.
    *   Uses the `http-bidir-comm` crate to establish a long-lived connection (based on HTTP long-polling or SSE) with the Hypervisor to receive scheduling commands (e.g., "execute," "wait").
*   **Execution Control (Trap & Wait)**:
    *   When an intercepted CUDA function is called, `cuda-limiter` pauses the thread's execution.
    *   It sends a "request to execute" signal to the Hypervisor, including information about the current process and pod.
    *   It then blocks and waits for the Hypervisor's response. Only after receiving an "allow execute" command does it call the original CUDA function, allowing the GPU operation to proceed.
*   **Shared Memory Access**:
    *   By attaching to the shared memory region created by the Hypervisor, `cuda-limiter` can access the current node-wide GPU status with near-zero overhead for local decision-making or data reporting.
*   **Environment Variable Configuration**:
    *   Relies on environment variables (e.g., `HYPERVISOR_IP`, `HYPERVISOR_PORT`, `POD_NAME`) to obtain the necessary runtime context.

#### 2.3. http-bidir-comm

This is a generic, HTTP-based bidirectional communication library used by both the Hypervisor and Cuda-Limiter. It abstracts the common pattern of client-server task requests and result reporting.

*   **Client (Cuda-Limiter side)**:
    *   Implements a `BlockingHttpClient`, suitable for use in injected, potentially non-async code environments.
    *   Pulls tasks/commands from the server via long-polling or Server-Sent Events (SSE).
*   **Server (Hypervisor side)**:
    *   Implemented using the `Poem` web framework to provide an asynchronous HTTP service.
    *   Maintains task queues for each client (worker) and handles result submissions from clients.

#### 2.4. Utils

A common utility library providing shared functionality across multiple crates:

*   **Logging**: Standardized logging facilities.
*   **Hooking**: Low-level wrappers for API interception.
*   **Shared Memory**: Wrappers for creating and accessing shared memory.
*   **Build Info**: Embeds version and build information at compile time.

### 3. Workflow (From Pod Launch to GPU Usage)

1.  **Pod Scheduling**: Kubernetes schedules a GPU-requesting pod to a node.
2.  **Hypervisor Awareness**: The Hypervisor on that node, watching the K8s API Server, learns about the new pod.
3.  **Process Launch & Injection**: The container within the pod starts the user application. The `LD_PRELOAD` environment variable ensures `cuda-limiter.so` is loaded before the application.
4.  **Limiter Initialization**:
    *   The `cuda-limiter` code is executed via its `ctor` (constructor).
    *   It reads environment variables like `POD_NAME` and `HYPERVISOR_IP`.
    *   It establishes HTTP communication with the Hypervisor and registers itself, reporting its PID and parent Pod name.
5.  **Hypervisor Worker Registration**: The Hypervisor receives the registration request, records the new worker process, and associates it with the corresponding pod.
6.  **API Interception**: `cuda-limiter` sets up its hooks for CUDA APIs.
7.  **GPU Operation Request**: The user application calls a CUDA function (e.g., to launch a kernel).
8.  **Trap & Wait**:
    *   The `cuda-limiter` interceptor catches the call, preventing its immediate execution.
    *   It sends an execution request to the Hypervisor via `http-bidir-comm`.
    *   The current thread is suspended, awaiting a scheduling decision from the Hypervisor.
9.  **Hypervisor Scheduling**:
    *   The Hypervisor's scheduler receives the request.
    *   Based on the GPU demand of all pods, their weights/quotas, and the current GPU state, it decides whether to grant the request.
    *   If granted, it sends an "allow execute" command back to the corresponding `cuda-limiter` over the HTTP connection.
10. **Execution & Resumption**:
    *   `cuda-limiter` receives the command and wakes up the suspended thread.
    *   The original CUDA function is called, and the actual GPU computation begins.
11. **Completion & Loop**: Once the GPU operation is complete, the function returns. `cuda-limiter` is ready to intercept the next CUDA call, repeating the process.
12. **Pod Termination**: When the pod is deleted, the Hypervisor cleans up all records and state associated with that pod's workers.

### 4. Architecture Diagram

```
+-------------------------------------------------------------------+
| Kubernetes Node                                                   |
|                                                                   |
|  +---------------------------+      +---------------------------+ |
|  | Pod A                     |      | Pod B                     | |
|  |                           |      |                           | |
|  | +-----------------------+ |      | +-----------------------+ | |
|  | | Process 1 (User App)  | |      | | Process 3 (User App)  | | |
|  | | +-------------------+ | |      | | +-------------------+ | | |
|  | | | cuda-limiter.so |<----HTTP---->| | cuda-limiter.so |<----...
|  | | +-------------------+ | |      | | +-------------------+ | | |
|  | +-----------------------+ |      | +-----------------------+ | |
|  |                           |      |                           | |
|  | +-----------------------+ |      +---------------------------+ |
|  | | Process 2 (User App)  | |                                    |
|  | | +-------------------+ | |                                    |
|  | | | cuda-limiter.so |<----HTTP---->+                           |
|  | | +-------------------+ | |      |                           |
|  | +-----------------------+ |      |                           |
|  +---------------------------+      |  Hypervisor (Daemon)      |
|                                     |                           |
|                                     | +-----------------------+ |
|       +---------------------------->| |    Scheduler          | |
|       |                             | +-----------------------+ |
|       |                             | |    Worker Manager     | |
|       |                             | +-----------------------+ |
|       |                             | |    K8s Pod Watcher    | |
|       |                             | +-----------------------+ |
|       |                             | |    HTTP API Server    | |
|       |                             | | (http-bidir-comm)   | |
|       |                             | +-----------------------+ |
|       |                             | |    GPU Observer (NVML)| |
|       +-----------------------------+-----------------------+ |
|       |                                                       |
|  +----|------------------------+      +-----------------------+ |
|  | Shared Memory (/dev/shm) |<----(write)----| |
|  +----|------------------------+      +-----------------------+ |
|       |      ^                                                  |
|       +------|--------------------------------------------------+
|              | (read)
|              |
|  +-----------+---------------+
|  | cuda-limiter.so (in any process) |
|  +---------------------------+
|                                                                   |
+-------------------------------------------------------------------+
| |<-- K8s API -->| Kubernetes API Server                           |
+-------------------------------------------------------------------+
```
