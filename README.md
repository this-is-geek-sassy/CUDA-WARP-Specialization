# CUDA-WARP-Specialization

This repository contains optimized CUDA kernel implementations that are ready to use with PolyBenchGPU.

## What's Inside

### GEMM-FP32

Single-precision matrix multiplication with optimized CUDA kernels including warp-specialized implementations using cudaDMA.

### GEMM-FP64

Double-precision matrix multiplication with optimized CUDA kernels featuring register tiling, double buffering, and warp specialization techniques.

### GEMM_TENSOR

Tensor Core-based GEMM implementation leveraging NVIDIA's Tensor Cores for high-performance matrix operations on Ampere+ architectures.

### 3DCONV-FP32

3D convolution kernels with various optimization strategies including shared memory tiling, async copy, and warp specialization.

### JACOBI2D-FP32

2D Jacobi iterative stencil computation with optimized memory access patterns and double buffering.

### STENCIL-FP32

General stencil computation kernels with optimized memory hierarchies.

> **Note:** Some kernels using cudaDMA may encounter illegal instruction errors on certain GPU architectures. See individual kernel folders for detailed documentation and workarounds.

Each folder is self-contained with its own `Makefile`, source files, and test cases.

## How to Use with PolyBenchGPU

Simply copy the kernel folders alongside `gpu_utils.h` in your `polybenchGPU/CUDA` directory:

```bash
# Navigate to your PolyBenchGPU CUDA folder
cd /path/to/polybenchGPU/CUDA

# Copy the kernel folders from this repository
cp -r /path/to/CUDA-WARP-Specialization/GEMM-FP32 .
cp -r /path/to/CUDA-WARP-Specialization/GEMM-FP64 .
cp -r /path/to/CUDA-WARP-Specialization/GEMM_TENSOR .
cp -r /path/to/CUDA-WARP-Specialization/3DCONV-FP32 .
cp -r /path/to/CUDA-WARP-Specialization/JACOBI2D-FP32 .
cp -r /path/to/CUDA-WARP-Specialization/STENCIL-FP32 .
# Copy the utility header
cp /path/to/CUDA-WARP-Specialization/gpu_utils.h .
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
├── GEMM_TENSOR/
│   ├── Makefile
│   └── ...
├── 3DCONV-FP32/
│   ├── Makefile
│   └── ...
├── JACOBI2D-FP32/
│   ├── Makefile
│   └── ...
├── STENCIL-FP32/
│   ├── Makefile
│   └── ...
└── [additional kernels]/
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

# Example: Build GEMM_TENSOR
cd GEMM_TENSOR
make

# Run with specific dataset size
./gemm_tensor

# Example: Build 3D Convolution
cd 3DCONV-FP32
make
./conv3d_baseline  # or other variants
```

The same pattern applies to all kernel implementations in this repository.

## Known Issues

### cudaDMA Illegal Instruction Error

Some kernels using the cudaDMA library (particularly in STENCIL-FP32 and JACOBI2D-FP32) may encounter illegal instruction errors when compiled with optimizations on certain GPU architectures. See `STENCIL-FP32/nvcc-glitch.md` and `STENCIL-FP32/cudadma-illegal-instruction-issue.tex` for detailed documentation.

**Quick workaround:** Compile with debug flags:

```bash
nvcc -G -g -O0 -arch=sm_70 your_file.cu
```

## Performance Analysis

Nsight Compute reports for GEMM, Jacobi2D, and 3D Convolution kernels are available in the respective `nsight report` directories for detailed performance analysis.
