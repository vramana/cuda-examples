#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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
    constexpr int numElements = 1 << 20;  // 1,048,576 elements
    constexpr size_t bytes = numElements * sizeof(float);

    // Allocate host memory and initialize input data.
    float* h_a = static_cast<float*>(std::malloc(bytes));
    float* h_b = static_cast<float*>(std::malloc(bytes));
    float* h_c = static_cast<float*>(std::malloc(bytes));

    if (!h_a || !h_b || !h_c) {
        std::fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < numElements; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(numElements - i);
    }

    // Allocate device memory.
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Copy input data to the GPU.
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel with one thread per element.
    constexpr int threadsPerBlock = 256;
    const int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddNaive<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, numElements);
    CUDA_CHECK(cudaGetLastError());

    // Copy the result back to host.
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify results.
    bool allOk = true;
    for (int i = 0; i < numElements; ++i) {
        const float expected = h_a[i] + h_b[i];
        if (std::abs(h_c[i] - expected) > 1e-5f) {
            std::fprintf(stderr, "Mismatch at %d: %f vs %f\n", i, h_c[i], expected);
            allOk = false;
            break;
        }
    }

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

    return allOk ? EXIT_SUCCESS : EXIT_FAILURE;
}
