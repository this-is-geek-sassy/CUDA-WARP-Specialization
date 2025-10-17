# CUDA-WARP-Specialization

This repository contains optimized CUDA kernel implementations that are ready to use with PolyBenchGPU.

## What's Inside

- **GEMM-FP64/** — Double-precision matrix multiplication with multiple kernel implementations
- **GEMM-TENSOR/** — Experimental tensor-based GEMM implementation
- _More kernels coming soon..._

Each folder is self-contained with its own `Makefile`, source files, and test cases.

## How to Use with PolyBenchGPU

Simply copy the kernel folders alongside `gpu_utils.h` in your `polybenchGPU/CUDA` directory:

```bash
# Navigate to your PolyBenchGPU CUDA folder
cd /path/to/polybenchGPU/CUDA

# Copy the kernel folders from this repository
cp -r /path/to/CUDA-WARP-Specialization/GEMM-FP64 .
cp -r /path/to/CUDA-WARP-Specialization/GEMM-TENSOR .
# Add more kernels as they become available...
```

Your directory structure should look like:

```text
polybenchGPU/CUDA/
├── gpu_utils.h
├── GEMM-FP64/
│   ├── Makefile
│   ├── main.cpp
│   ├── drivers/
│   ├── kernels/
│   └── test_cases/
├── GEMM-TENSOR/
│   └── ...
└── [other kernel folders]/
    └── ...
```

## Building and Running

Each kernel folder has its own Makefile. Navigate to any kernel folder and run `make`:

```bash
# Example: Build GEMM-FP64
cd GEMM-FP64
make

# Run with test case 1, kernel 1 (basic implementation)
./bin/dgemm 1 1

# Run with test case 1, kernel 2 (2D tiled implementation)
./bin/dgemm 1 2
```

The same pattern applies to all kernel implementations in this repository.
