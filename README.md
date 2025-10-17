# CUDA-WARP-Specialization

This repository contains optimized CUDA kernel implementations that are ready to use with PolyBenchGPU.

## What's Inside

### GEMM-FP32

Single-precision matrix multiplication with optimized CUDA kernels.

### GEMM-FP64

Double-precision matrix multiplication with optimized CUDA kernels.

### GEMM-TENSOR

Tensor Core-based GEMM implementation for high-performance matrix operations.

_More kernels coming soon..._

Each folder is self-contained with its own `Makefile`, source files, and test cases.

## How to Use with PolyBenchGPU

Simply copy the kernel folders alongside `gpu_utils.h` in your `polybenchGPU/CUDA` directory:

```bash
# Navigate to your PolyBenchGPU CUDA folder
cd /path/to/polybenchGPU/CUDA

# Copy the kernel folders from this repository
cp -r /path/to/CUDA-WARP-Specialization/GEMM-FP32 .
cp -r /path/to/CUDA-WARP-Specialization/GEMM-FP64 .
cp -r /path/to/CUDA-WARP-Specialization/GEMM-TENSOR .
# Add more kernels as they become available...
```

Your directory structure should look like:

```text
polybenchGPU/CUDA/
├── gpu_utils.h
├── GEMM-FP32/
│   ├── Makefile
│   └── ...
├── GEMM-FP64/
│   ├── Makefile
│   └── ...
├── GEMM-TENSOR/
│   ├── Makefile
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

# Run the executable (check the specific folder's README for usage)
./gemm_fp64

# Example: Build GEMM-TENSOR
cd GEMM-TENSOR
make

# Run with specific dataset size
./gemm_tensor
```

The same pattern applies to all kernel implementations in this repository.
