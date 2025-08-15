#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    printf("  --gpu-index <n>          GPU index (default: 0)\n");
    printf("  --iterations <n>         Number of kernel launches (default: 100)\n");
    printf("  --help                   Show this help\n");
}

// Function declarations
int run_iterations(size_t memory_bytes, long iterations);

int main(int argc, char **argv) {
    // Default values
    size_t memory_bytes = 256 * 1024 * 1024; // 256MB
    int gpu_index = 0;
    long iterations = 100; // Default number of iterations
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--memory-bytes") == 0 && i + 1 < argc) {
            memory_bytes = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-index") == 0 && i + 1 < argc) {
            gpu_index = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atol(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    printf("Starting CUDA test program with the following parameters:\n");
    printf("  Memory: %zu bytes\n", memory_bytes);
    printf("  GPU: %d\n", gpu_index);
    printf("  Iterations: %ld\n", iterations);
    
    // set device
    cudaError_t err = cudaSetDevice(gpu_index);
    if (err != cudaSuccess) {
        printf("Failed to set CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    return run_iterations(memory_bytes, iterations);
}



int run_iterations(size_t memory_bytes, long iterations) {
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
