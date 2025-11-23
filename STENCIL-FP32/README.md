# Jacobi 2D Stencil - Warp Specialization

CUDA implementation of the Jacobi 2D iterative stencil computation with multiple optimization strategies including warp-specialized memory management.

## Overview

This project implements a 5-point stencil Jacobi iterative solver on 2D grids using multiple CUDA kernel variants:

### Baseline Implementation (`jacobi2D_baseline.cu`)

1. **Baseline (No Shared Memory)**: Simple GPU kernel with direct global memory access
2. **Shared Memory Optimized**: Kernel using shared memory with halo regions for reduced global memory access
3. **Texture Memory**: Kernel leveraging CUDA texture memory for cached reads
4. **Texture + Shared Memory Hybrid**: Combined approach using both texture and shared memory optimizations

### Warp-Specialized Implementation (`jacobi2D_cudaDMA.cu`)

1. **cp.async Warp-Specialized**: Advanced kernel using asynchronous copy instructions (cp.async PTX) to separate compute and memory transfer threads
2. **Pure cudaDMA Library** ⚠️ **[DEPRECATED/BROKEN]**: Attempts to use the cudaDMA library but **fails with PTX error "an illegal instruction was encountered"** on modern architectures (sm_86+). The cudaDMA library contains old inline PTX assembly incompatible with Ampere and newer GPUs.

## Features

- Multiple dataset sizes (128² to 16384²)
- Two separate implementations with different kernel variants
- Automated benchmarking scripts for both baseline and warp-specialized versions
- CPU reference implementation for validation
- Performance comparison and speedup calculations
- Detailed timing instrumentation with 5-run averaging

## Building

### Baseline Implementation

```bash
# Compile baseline with all 4 kernel variants
make DATASET=STANDARD_DATASET

# Or using nvcc directly
nvcc -O3 -arch=sm_86 -DSTANDARD_DATASET -I../.. jacobi2D_baseline.cu -o jacobi2D_baseline
```

### Warp-Specialized Implementation

```bash
# Compile cudaDMA version (uses cp.async kernel)
make -f Makefile_cudaDMA DATASET=STANDARD_DATASET

# Or using nvcc directly
nvcc -O3 -arch=sm_86 -DSTANDARD_DATASET -I../.. jacobi2D_cudaDMA.cu -o jacobi2D_cudaDMA
```

### Available Dataset Sizes

```bash
# -DMINI_DATASET        (128x128, 20 timesteps)
# -DSMALL_DATASET       (256x256, 40 timesteps)
# -DSTANDARD_DATASET    (1024x1024, 100 timesteps)
# -DLARGE_DATASET       (2048x2048, 200 timesteps)
# -DEXTRALARGE_DATASET  (4096x4096, 500 timesteps)
```

## Running

### Single Execution

```bash
# Run baseline implementation (all 4 kernel variants)
./jacobi2D_baseline

# Run warp-specialized implementation (cp.async kernel)
./jacobi2D_cudaDMA
```

### Automated Benchmarks

#### Baseline Benchmark

```bash
# Run benchmark for baseline kernels - all sizes up to EXTRALARGE
./benchmark_baseline.sh

# Run with size limit (e.g., up to 2048x2048)
./benchmark_baseline.sh 2048
```

#### Warp-Specialized Benchmark

```bash
# Run benchmark for cudaDMA kernels (baseline, shared, cudaDMA cp.async)
./benchmark.sh

# Run benchmark with size limit (e.g., up to 1024x1024)
./benchmark.sh 1024
```

Both benchmark scripts:

- Test multiple dataset sizes automatically
- Run each configuration 5 times for statistical reliability
- Calculate average execution times
- Compute speedups relative to baseline kernel
- Generate timestamped log files with results
- Validate all kernels for correctness (exits on mismatch)

## Performance

### Baseline Kernels

Typical speedups observed relative to baseline (no shared memory):

- **Shared Memory**: 1.3-1.4x improvement
- **Texture Memory**: 1.2-1.3x improvement
- **Texture + Shared Hybrid**: 1.3x improvement

### Warp-Specialized Kernels

The cp.async warp-specialized variant shows:

- Competitive performance with shared memory optimized kernels
- Better performance at larger problem sizes
- **0 validation errors** - produces correct results

### GPU vs CPU Speedup

Observed speedups (GPU over CPU):

- 1024×1024: ~3-4x
- 2048×2048: ~6x
- 4096×4096: ~12x

### ⚠️ cudaDMA Library Issue

The pure cudaDMA library implementation (using cudaDMAStrided class) **cannot be used on modern GPUs**:

- **Error**: "an illegal instruction was encountered"
- **Cause**: Old inline PTX assembly in cudaDMA library incompatible with sm_86+ architectures
- **Workaround**: Use cp.async kernel variant instead (fully functional)

## Algorithm

The Jacobi 2D stencil updates each grid point using its 4 neighbors:

```
B[i][j] = 0.2 * (A[i][j] + A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1])
```

The computation is iterated for multiple timesteps, swapping input/output arrays between iterations.

## Implementation Notes

### Baseline Kernels

- **No Shared Memory**: Direct global memory access for simplicity
- **Shared Memory**: 32×8 thread blocks with 2-element halo loading
- **Texture Memory**: Uses texture cache for read-only input array
- **Hybrid**: Combines texture reads with shared memory tiling

### Warp-Specialized (cp.async)

- **Thread Configuration**: 900 compute threads (30×30) + 32 DMA threads = 932 total threads/block
- **Tile Size**: 32×32 with 1-element halo border (30×30 compute region)
- **Memory Pattern**: Asynchronous cooperative loading using cp.async PTX instructions
- **Synchronization**: Separate compute and memory transfer warps with explicit barriers
- **Validation**: Achieves 0 errors, matching CPU reference implementation

### Legacy cudaDMA Library Issues

- Located in `cudaDMA.h` and `cudaDMAv2.h`
- Uses `cudaDMAStrided<>` template class for warp specialization
- **Status**: BROKEN on modern architectures
- **Error Message**: `CUDA Error: an illegal instruction was encountered`
- **Root Cause**: Inline PTX assembly incompatible with Ampere (sm_86) and newer
- **Alternative**: Use `jacobi2D_kernel_cudaDMA` (cp.async version) instead of `jacobi2D_kernel_pure_cudaDMA`

## Requirements

- CUDA Toolkit 11.0+
- GPU with compute capability 8.6+ (Ampere architecture)
- Polybench GPU utilities

## Output

### Baseline Execution

Each execution reports:

- GPU execution time for all 4 kernel variants (Baseline, Shared, Texture, Hybrid)
- CPU execution time
- Validation results for each variant (number of mismatches)

### Warp-Specialized Execution

Each execution reports:

- GPU execution time for all 3 kernels (Baseline, Shared Memory, cudaDMA cp.async)
- CPU execution time
- Validation results (number of mismatches)
- CUDA error detection after kernel launches

### Benchmark Logs

Benchmark runs generate timestamped log files with detailed statistics:

- `benchmark_baseline_results_YYYYMMDD_HHMMSS.txt` - Baseline kernel benchmarks
- `benchmark_results_YYYYMMDD_HHMMSS.txt` - Warp-specialized kernel benchmarks
