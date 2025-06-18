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

// Memory allocation patterns
typedef enum {
    PATTERN_SINGLE,
    PATTERN_REPEATED, 
    PATTERN_GRADUAL
} AllocationPattern;

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --memory-bytes <size>    Memory size in bytes\n");
    printf("  --duration <seconds>     Duration in seconds\n");
    printf("  --pattern <type>         Allocation pattern: single, repeated, gradual\n");
    printf("  --count <n>              Count for repeated pattern\n");
    printf("  --step <size>            Step size for gradual pattern\n");
    printf("  --max <size>             Max size for gradual pattern\n");
    printf("  --gpu-index <n>          GPU index (default: 0)\n");
    printf("  --help                   Show this help\n");
}

// Function declarations
int run_single_allocation(size_t memory_bytes, time_t start_time, time_t end_time);
int run_repeated_allocation(size_t alloc_size, int count, time_t start_time, time_t end_time);
int run_gradual_allocation(size_t start_size, size_t step, size_t max_size, time_t start_time, time_t end_time);

int main(int argc, char **argv) {
    // Default values
    size_t memory_bytes = 256 * 1024 * 1024; // 256MB
    int duration_seconds = 10;
    int gpu_index = 0;
    AllocationPattern pattern = PATTERN_SINGLE;
    int count = 1;
    size_t step = 64 * 1024 * 1024; // 64MB
    size_t max_size = 512 * 1024 * 1024; // 512MB
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--memory-bytes") == 0 && i + 1 < argc) {
            memory_bytes = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            duration_seconds = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pattern") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "single") == 0) {
                pattern = PATTERN_SINGLE;
            } else if (strcmp(argv[i], "repeated") == 0) {
                pattern = PATTERN_REPEATED;
            } else if (strcmp(argv[i], "gradual") == 0) {
                pattern = PATTERN_GRADUAL;
            } else {
                printf("Unknown pattern: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--count") == 0 && i + 1 < argc) {
            count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--step") == 0 && i + 1 < argc) {
            step = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            max_size = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-index") == 0 && i + 1 < argc) {
            gpu_index = atoi(argv[++i]);
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
    printf("  Pattern: %s\n", pattern == PATTERN_SINGLE ? "single" : 
                              pattern == PATTERN_REPEATED ? "repeated" : "gradual");
    
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
    
    switch (pattern) {
        case PATTERN_SINGLE:
            result = run_single_allocation(memory_bytes, start_time, end_time);
            break;
        case PATTERN_REPEATED:
            result = run_repeated_allocation(memory_bytes, count, start_time, end_time);
            break;
        case PATTERN_GRADUAL:
            result = run_gradual_allocation(memory_bytes, step, max_size, start_time, end_time);
            break;
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

int run_repeated_allocation(size_t alloc_size, int count, time_t start_time, time_t end_time) {
    printf("Starting repeated allocation: %d allocations of %zu bytes each\n", count, alloc_size);
    
    float **allocations = (float**)malloc(count * sizeof(float*));
    if (!allocations) {
        printf("Failed to allocate allocation tracking array\n");
        return 1;
    }
    
    // Initialize array
    for (int i = 0; i < count; i++) {
        allocations[i] = NULL;
    }
    
    int successful_allocations = 0;
    
    // Try to allocate all requested memory blocks
    for (int i = 0; i < count && time(NULL) < end_time; i++) {
        printf("Allocation %d/%d: %zu bytes\n", i + 1, count, alloc_size);
        
        cudaError_t err = cudaMalloc((void**)&allocations[i], alloc_size);
        if (err != cudaSuccess) {
            printf("Failed allocation %d: %s\n", i + 1, cudaGetErrorString(err));
            if (err == cudaErrorMemoryAllocation) {
                printf("out of memory\n");
            }
            break;
        }
        successful_allocations++;
        
        // Sleep a bit between allocations
        sleep(1);
    }
    
    printf("Successfully allocated %d/%d blocks\n", successful_allocations, count);
    
    // Keep the memory allocated and do some work
    while (time(NULL) < end_time) {
        sleep(1);
    }
    
    // Clean up
    for (int i = 0; i < successful_allocations; i++) {
        if (allocations[i]) {
            cudaFree(allocations[i]);
        }
    }
    free(allocations);
    
    printf("Repeated allocation test completed\n");
    return 0;
}

int run_gradual_allocation(size_t start_size, size_t step, size_t max_size, time_t start_time, time_t end_time) {
    printf("Starting gradual allocation: %zu bytes -> %zu bytes (step: %zu)\n", 
           start_size, max_size, step);
    
    size_t current_size = start_size;
    float *d_data = NULL;
    
    while (current_size <= max_size && time(NULL) < end_time) {
        printf("Allocating %zu bytes\n", current_size);
        
        // Free previous allocation
        if (d_data) {
            cudaFree(d_data);
            d_data = NULL;
        }
        
        cudaError_t err = cudaMalloc((void**)&d_data, current_size);
        if (err != cudaSuccess) {
            printf("Failed to allocate %zu bytes: %s\n", current_size, cudaGetErrorString(err));
            if (err == cudaErrorMemoryAllocation) {
                printf("out of memory\n");
            }
            break;
        }
        
        printf("Successfully allocated %zu bytes\n", current_size);
        
        // Do some work with the allocated memory
        sleep(2);
        
        current_size += step;
    }
    
    // Keep final allocation and do work
    while (time(NULL) < end_time) {
        sleep(1);
    }
    
    // Clean up
    if (d_data) {
        cudaFree(d_data);
    }
    
    printf("Gradual allocation test completed\n");
    return 0;
}
