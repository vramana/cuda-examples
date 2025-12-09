#include <cmath>
#include <vector>
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
    std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096, 8192};
    int m, n, k, max_size;
    max_size = SIZE[SIZE.size() - 1];

    size_t bytes = max_size * sizeof(float);

    std::cout << "Max: " << max_size << std::endl;

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

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    randomize_vector(h_a, max_size);
    randomize_vector(h_b, max_size);
    randomize_vector(h_c, max_size);
    randomize_vector(h_d, max_size);

    std::printf("initialization complete\n");

    // Allocate device memory.
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_d = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_d, bytes));

    // Copy input data to the GPU.
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));


    std::printf("launch kernels\n");

    // Launch kernel with one thread per element.

    int repeat_times = 50;
    std::vector<float> timings = {};

    for (int size: SIZE) {
      run_vector_add_naive(d_a, d_b, d_c, SIZE[0]);
      run_vector_add_cublas(handle, d_a, d_b, d_d, SIZE[0]);

      std::printf("completed kernels\n");

      // Copy the result back to host.
      CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_d, d_d, bytes, cudaMemcpyDeviceToHost));

      std::printf("verify results\n");

      // Verify results.
      bool allOk = compare_vectors(h_c, h_d, SIZE[0]);

      if (allOk) {
          std::printf("Vector addition was successful!\n");
      }

      cudaEventRecord(beg);
      for (int i = 0; i < repeat_times; i++) {
        // run_vector_add_naive(d_a, d_b, d_c, size);
        run_vector_add_cublas(handle, d_a, d_b, d_c, size);
      }
      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);

      float elapsed_time = 0;

      cudaEventElapsedTime(&elapsed_time, beg, end);

      timings.push_back(elapsed_time / 1000.0);

      printf("size: %d, timing: %f\n", size, elapsed_time / 1000.0);
    }

    // Cleanup.
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_d));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
    std::free(h_d);

    cublasDestroy(handle);

    return 0;
}
