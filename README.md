# CUDA Vector Addition Example

This repository contains a minimal CUDA program (`main.cu`) that performs vector addition on the GPU and verifies the results on the host. You need the NVIDIA CUDA Toolkit installed so that `nvcc` and the CUDA runtime headers are available.

## Build and Run

```bash
nvcc -O2 -o vector_add main.cu
./vector_add
```

If compilation succeeds, the program prints `Vector addition was successful!` after validating the GPU-computed results.
