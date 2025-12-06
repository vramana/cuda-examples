#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "add_naive.cuh"

// Simple CUDA error checking macro to keep the example readable.
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(err), __FILE__, __LINE__);          \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

int main() {
    constexpr int numElements = 1 << 10;  // 1,048,576 elements
    constexpr size_t bytes = numElements * sizeof(float);

    cublasHandle_t handle;

    cublasCreate(&handle);

    // Allocate host memory and initialize input data.
    float* h_a = static_cast<float*>(std::malloc(bytes));
    float* h_b = static_cast<float*>(std::malloc(bytes));
    float* h_c = static_cast<float*>(std::malloc(bytes));
    float* h_d = static_cast<float*>(std::malloc(bytes));

    if (!h_a || !h_b || !h_c) {
        std::fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    randomize_vector(h_a, numElements);
    randomize_vector(h_b, numElements);
    zero_init_vector(h_c, numElements);
    zero_init_vector(h_d, numElements);

    // Allocate device memory.
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_d = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_d, bytes));

    // Copy input data to the GPU.
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel with one thread per element.
    run_vector_add_naive(d_a, d_b, d_c, numElements);
    run_vector_add_cublas(&handle, d_a, d_b, d_d, numElements);

    // Copy the result back to host.
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_d, d_d, bytes, cudaMemcpyDeviceToHost));

    // Verify results.
    bool allOk = compare_vectors(d_c, d_d, numElements);

    if (allOk) {
        std::printf("Vector addition was successful!\n");
    }

    // Cleanup.
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);


    cublasDestroy(handle);

    return allOk ? EXIT_SUCCESS : EXIT_FAILURE;
}
