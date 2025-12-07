#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

__global__ void vector_add_native(const float* a, const float* b, float* c, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void randomize_vector(float *v, int n) {
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);

  for (int i = 0; i < n; i++) {
    float tmp = float(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (1.0);
    v[i] = n;
  }
}

void zero_init_vector(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}


bool compare_vectors(const float *a, const float *b, int n) {
  for (int i = 0; i < n; i++) {
    std::printf("Comparing %d element: %f, %f", i, a[i], b[i]);
    if (std::abs(a[i]-b[i]) > 0.001) {
      return false;
    }
  }

  return true;
}


void run_vector_add_naive(const float* d_a, const float* d_b, float* d_c, int n) {
    constexpr int threadsPerBlock = 32;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_native<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s at %s:%d\n",
                     cudaGetErrorString(err), __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }
}


void run_vector_add_cublas(cublasHandle_t handle, const float* d_a, const float* d_b, float* d_c, int n) {
  const float alpha = 1.0;
  cublasSaxpy(handle, n, &alpha, d_a, 1, d_c, 1);
  cublasSaxpy(handle, n, &alpha, d_b, 1, d_c, 1);
}
