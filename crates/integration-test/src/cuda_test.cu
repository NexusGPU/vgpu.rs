#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>

// simple compute kernel to generate GPU load
__global__ void computeKernel(float *input, float *output, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        
        // execute some compute to generate GPU load
        for (int i = 0; i < iterations; i++) {
            val = val + 0.01f;
            val = val * 1.01f;
            val = val + __sinf(val);
        }
        
        output[idx] = val;
    }
}

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --memory-bytes <size>    Memory size in bytes\n");
    printf("  --duration <seconds>     Duration in seconds\n");
    printf("  --gpu-index <n>          GPU index (default: 0)\n");
    printf("  --fixed-iterations <n>   Run a fixed number of kernel launches (preferred)\n");
    printf("  --help                   Show this help\n");
}

// Function declarations
int run_single_allocation(size_t memory_bytes, time_t start_time, time_t end_time);
int run_fixed_iterations(size_t memory_bytes, long iterations);

int main(int argc, char **argv) {
    // Default values
    size_t memory_bytes = 256 * 1024 * 1024; // 256MB
    int duration_seconds = 10;
    int gpu_index = 0;
    // Preferred verification path
    long fixed_iterations = -1; // if > 0, run fixed workload instead of duration-based loops
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--memory-bytes") == 0 && i + 1 < argc) {
            memory_bytes = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            duration_seconds = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-index") == 0 && i + 1 < argc) {
            gpu_index = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fixed-iterations") == 0 && i + 1 < argc) {
            fixed_iterations = atol(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (argc == 4 && i == 1) {
            // Legacy format: memory_bytes duration_seconds gpu_index
            memory_bytes = atoll(argv[1]);
            duration_seconds = atoi(argv[2]);
            gpu_index = atoi(argv[3]);
            break;
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    printf("Starting CUDA test program with the following parameters:\n");
    printf("  Memory: %zu bytes\n", memory_bytes);
    printf("  Duration: %d seconds\n", duration_seconds);
    printf("  GPU: %d\n", gpu_index);
    if (fixed_iterations > 0) {
        printf("  Mode: fixed-iterations (%ld)\n", fixed_iterations);
    } else {
        printf("  Mode: duration-based (single)\n");
    }
    
    // set device
    cudaError_t err = cudaSetDevice(gpu_index);
    if (err != cudaSuccess) {
        printf("Failed to set CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // record start time
    time_t start_time = time(NULL);
    time_t end_time = start_time + duration_seconds;
    
    int result = 0;
    
    if (fixed_iterations > 0) {
        result = run_fixed_iterations(memory_bytes, fixed_iterations);
    } else {
        result = run_single_allocation(memory_bytes, start_time, end_time);
    }
    
    return result;
}

int run_single_allocation(size_t memory_bytes, time_t start_time, time_t end_time) {
    // calculate the number of elements to allocate (float = 4 bytes)
    size_t element_count = memory_bytes / sizeof(float);
    size_t total_size = element_count * sizeof(float);
    
    // allocate and initialize host memory
    float *h_data = (float*)malloc(total_size);
    if (!h_data) {
        printf("Failed to allocate host memory\n");
        return 1;
    }
    
    // initialize input data
    for (size_t i = 0; i < element_count; i++) {
        h_data[i] = 1.0f;
    }
    
    // allocate GPU memory
    float *d_data = NULL;
    float *d_output = NULL;
    
    printf("Allocating %zu bytes of GPU memory\n", total_size);
    cudaError_t err = cudaMalloc((void**)&d_data, total_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for input: %s\n", cudaGetErrorString(err));
        if (err == cudaErrorMemoryAllocation) {
            printf("out of memory\n");
        }
        free(h_data);
        return 1;
    }
    
    err = cudaMalloc((void**)&d_output, total_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for output: %s\n", cudaGetErrorString(err));
        if (err == cudaErrorMemoryAllocation) {
            printf("out of memory\n");
        }
        cudaFree(d_data);
        free(h_data);
        return 1;
    }
    
    // copy data from host to device
    err = cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy data from host to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        cudaFree(d_data);
        free(h_data);
        return 1;
    }
    
    printf("Memory allocation complete\n");
    printf("Starting GPU workload for %ld seconds\n", end_time - start_time);
    
    // Configure CUDA kernel execution
    int threadsPerBlock = 256;
    int blocks = (element_count + threadsPerBlock - 1) / threadsPerBlock;
    
    // run until duration is reached
    while (time(NULL) < end_time) {
        // use a fixed high iteration count for maximum GPU utilization
        const int iterations = 1000; // high value to ensure full GPU utilization
        
        // launch kernel
        computeKernel<<<blocks, threadsPerBlock>>>(d_data, d_output, element_count, iterations);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to launch kernel: %s\n", cudaGetErrorString(err));
            break;
        }
        
        // wait for kernel to complete
        cudaDeviceSynchronize();
        // no sleep - run at maximum speed to fully utilize the GPU
    }
    
    printf("Test completed successfully\n");
    
    // clean up resources
    cudaFree(d_output);
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}

int run_fixed_iterations(size_t memory_bytes, long iterations) {
    // calculate the number of elements to allocate (float = 4 bytes)
    size_t element_count = memory_bytes / sizeof(float);
    size_t total_size = element_count * sizeof(float);

    // allocate and initialize host memory
    float *h_data = (float*)malloc(total_size);
    if (!h_data) {
        printf("Failed to allocate host memory\n");
        return 1;
    }

    for (size_t i = 0; i < element_count; i++) {
        h_data[i] = 1.0f;
    }

    // allocate GPU memory
    float *d_data = NULL;
    float *d_output = NULL;

    printf("[fixed] Allocating %zu bytes of GPU memory\n", total_size);
    cudaError_t err = cudaMalloc((void**)&d_data, total_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for input: %s\n", cudaGetErrorString(err));
        if (err == cudaErrorMemoryAllocation) {
            printf("out of memory\n");
        }
        free(h_data);
        return 1;
    }

    err = cudaMalloc((void**)&d_output, total_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for output: %s\n", cudaGetErrorString(err));
        if (err == cudaErrorMemoryAllocation) {
            printf("out of memory\n");
        }
        cudaFree(d_data);
        free(h_data);
        return 1;
    }

    // copy data from host to device
    err = cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy data from host to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        cudaFree(d_data);
        free(h_data);
        return 1;
    }

    printf("[fixed] Starting GPU workload for %ld iterations\n", iterations);

    // Configure CUDA kernel execution
    int threadsPerBlock = 256;
    int blocks = (element_count + threadsPerBlock - 1) / threadsPerBlock;

    const int inner_iterations = 1000; // keep kernel heavy
    for (long i = 0; i < iterations; i++) {
        computeKernel<<<blocks, threadsPerBlock>>>(d_data, d_output, element_count, inner_iterations);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to launch kernel: %s\n", cudaGetErrorString(err));
            break;
        }
        cudaDeviceSynchronize();
    }

    printf("[fixed] Fixed workload completed\n");

    // clean up resources
    cudaFree(d_output);
    cudaFree(d_data);
    free(h_data);

    return 0;
}

// Removed other experimental patterns; not needed for limiter validation
