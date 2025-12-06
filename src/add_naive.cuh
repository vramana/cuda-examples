#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vectorAddNaive(const float* a, const float* b, float* c, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

