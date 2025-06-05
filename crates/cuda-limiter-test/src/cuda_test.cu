#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
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

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: %s <memory_bytes> <utilization_percent> <duration_seconds> <gpu_index>\n", argv[0]);
        return 1;
    }
    
    size_t memory_bytes = atoll(argv[1]);
    int utilization = atoi(argv[2]);
    int duration_seconds = atoi(argv[3]);
    int gpu_index = atoi(argv[4]);
    
    printf("Starting CUDA test program with the following parameters:\n");
    printf("  Memory: %zu bytes\n", memory_bytes);
    printf("  Utilization: %d%%\n", utilization);
    printf("  Duration: %d seconds\n", duration_seconds);
    printf("  GPU: %d\n", gpu_index);
    
    // set device
    cudaError_t err = cudaSetDevice(gpu_index);
    if (err != cudaSuccess) {
        printf("Failed to set CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
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
    err = cudaMalloc((void**)&d_data, total_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for input: %s\n", cudaGetErrorString(err));
        free(h_data);
        return 1;
    }
    
    err = cudaMalloc((void**)&d_output, total_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for output: %s\n", cudaGetErrorString(err));
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
    printf("Starting GPU workload for %d seconds\n", duration_seconds);
    
    // Configure CUDA kernel execution
    int threadsPerBlock = 256;
    int blocks = (element_count + threadsPerBlock - 1) / threadsPerBlock;
    
    // record start time
    time_t start_time = time(NULL);
    time_t end_time = start_time + duration_seconds;
    
    // run until duration is reached
    while (time(NULL) < end_time) {
        // calculate iterations based on target utilization
        int iterations = utilization * 10; // scale factor
        
        // launch kernel
        computeKernel<<<blocks, threadsPerBlock>>>(d_data, d_output, element_count, iterations);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to launch kernel: %s\n", cudaGetErrorString(err));
            break;
        }
        
        // wait for kernel to complete
        cudaDeviceSynchronize();
        
        // adjust sleep time based on target utilization
        int adjustment_sleep = (100 - utilization) / 10;
        if (adjustment_sleep > 0) {
            usleep(adjustment_sleep * 1000); // milliseconds to microseconds
        }
    }
    
    printf("Test completed successfully\n");
    
    // clean up resources
    cudaFree(d_output);
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
