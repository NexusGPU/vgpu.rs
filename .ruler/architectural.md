# Architecture

## Project Overview & Core Architecture

You are developing a Kubernetes-based solution for GPU virtualization and resource limiting, named `vgpu.rs`.

### Core Components:

1.  **`hypervisor` (Daemon - `crates/hypervisor`):**
    *   **Role**: The central manager on a node, running as a daemon.
    *   **Responsibilities**:
        *   **Worker Management**: Manages all worker processes connected via `cuda-limiter`.
        *   **Kubernetes Integration**: Watches Pod events to map worker processes to their parent Pods, enabling a Pod-level view of resources.
        *   **GPU Monitoring**: Uses the `nvml-wrapper` crate to monitor physical GPU status.
        *   **Scheduling**: Implements a scheduler (e.g., `crates/hypervisor/src/scheduler/weighted.rs`) to decide which worker can execute GPU tasks based on pod weights and requests.
        *   **Communication**:
            *   Acts as an HTTP server using the `http-bidir-comm` crate for bidirectional communication with `cuda-limiter`.
            *   Creates and writes to a shared memory segment to expose global GPU state (like utilization) to all `cuda-limiter` instances.
    *   **Tech Stack**: `tokio`, `kube-rs`, `poem`, `nvml-wrapper`, `shared_memory`.

2.  **`cuda-limiter` (Injected Library - `crates/cuda-limiter`):**
    *   **Role**: A dynamic library (`.so`) injected into user GPU applications.
    *   **Responsibilities**:
        *   **API Interception (Hooking)**: Uses `frida-gum` or similar mechanisms to intercept critical CUDA APIs (e.g., `cuLaunchKernel`) and NVML APIs.
        *   **Communication**:
            *   Acts as an HTTP client using `http-bidir-comm`'s `BlockingHttpClient` to communicate with the `hypervisor`.
            *   Registers with the `hypervisor` on initialization, reporting its PID and the `POD_NAME` obtained from an environment variable.
            *   Receives execute/wait commands from the `hypervisor`.
        *   **Execution Control (Trap)**: When a GPU call is intercepted, it pauses the current thread, requests permission to execute from the `hypervisor`, and waits for its response.
        *   **Shared Memory**: Reads the shared memory created by the `hypervisor` to get real-time global GPU status.
    *   **Tech Stack**: `libloading`, `frida-gum`, `reqwest` (via `http-bidir-comm`), `shared_memory`, `ctor`.

3.  **`http-bidir-comm` (Communication Library - `crates/http-bidir-comm`):**
    *   **Role**: Provides an abstraction for bidirectional communication between the `hypervisor` and `cuda-limiter`.
    *   **Implementation**:
        *   **Server (`HttpServer`)**: Used in the `hypervisor`, based on `poem`, manages task queues.
        *   **Client (`BlockingHttpClient`)**: Used in `cuda-limiter`, based on `reqwest::blocking`, fetches tasks via long-polling or SSE.

### Key Design Patterns & Constraints:

*   **Pod-Level Limiting**: All scheduling and limiting are performed at the Pod level. All processes within the same Pod share a GPU quota. The `hypervisor` must maintain a mapping from `worker PID` to `Pod`.
*   **Environment-Driven Configuration**: `cuda-limiter` heavily relies on environment variables (`HYPERVISOR_IP`, `HYPERVISOR_PORT`, `POD_NAME`, `NVIDIA_VISIBLE_DEVICES`) for its initialization.
*   **Intercept & Block**: The core of `cuda-limiter` is the "intercept-request-block-wait-execute" loop. This is critical for implementing scheduling.
*   **Decoupling**: The `hypervisor` is agnostic to the user process's logic; it only schedules. The `cuda-limiter` is agnostic to the scheduling policy; it only executes the `hypervisor`'s commands.
*   **Async vs. Blocking**: The `hypervisor` is fully asynchronous (`tokio`). `cuda-limiter`, being injected into potentially non-async user code, must use blocking communication with the `hypervisor`.
