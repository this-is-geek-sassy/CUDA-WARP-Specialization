# Jacobi 2D Stencil - Warp Specialization

CUDA implementation of the Jacobi 2D iterative stencil computation with multiple optimization strategies including warp-specialized memory management.

## Overview

This project implements a 5-point stencil Jacobi iterative solver on 2D grids using three different CUDA kernel variants:

1. **Baseline**: Simple GPU kernel without shared memory
2. **Shared Memory**: Optimized kernel using shared memory with halo regions
3. **cudaDMA Warp-Specialized**: Advanced kernel separating compute and memory transfer warps

## Features

- Multiple dataset sizes (128² to 16384²)
- Automated benchmarking with 5-run averaging
- CPU reference implementation for validation
- Performance comparison and speedup calculations
- Detailed timing instrumentation

## Building

```bash
# Compile for specific dataset size
nvcc -O3 -arch=sm_86 -DSTANDARD_DATASET -I../.. jacobi2D_cudaDMA.cu -o jacobi2D_cudaDMA

# Available dataset sizes:
# -DMINI_DATASET        (128x128, 20 timesteps)
# -DSMALL_DATASET       (256x256, 40 timesteps)
# -DSTANDARD_DATASET    (1024x1024, 100 timesteps)
# -DLARGE_DATASET       (2048x2048, 200 timesteps)
# -DEXTRALARGE_DATASET  (4096x4096, 500 timesteps)
```

## Running

### Single Execution

```bash
./jacobi2D_cudaDMA
```

### Automated Benchmark

```bash
# Run benchmark for all sizes up to EXTRALARGE
./benchmark.sh

# Run benchmark with size limit (e.g., up to 2048x2048)
./benchmark.sh 2048
```

The benchmark script:

- Tests multiple dataset sizes automatically
- Runs each configuration 5 times for statistical reliability
- Calculates average execution times
- Computes speedups relative to baseline kernel
- Generates timestamped log files with results
- Validates all kernels for correctness (exits on mismatch)

## Performance

Typical speedups observed (GPU over CPU):

- 1024×1024: ~3-4x
- 2048×2048: ~6x
- 4096×4096: ~12x

The cudaDMA warp-specialized variant shows competitive or better performance compared to the baseline, especially at larger problem sizes.

## Algorithm

The Jacobi 2D stencil updates each grid point using its 4 neighbors:

```
B[i][j] = 0.2 * (A[i][j] + A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1])
```

The computation is iterated for multiple timesteps, swapping input/output arrays between iterations.

## Implementation Notes

- **Thread Configuration**: 256 compute threads + 32 DMA threads per block
- **Tile Size**: 32×32 with 1-element halo border (30×30 compute region)
- **Memory Pattern**: Cooperative loading in warp-specialized variant
- **Validation**: Checks CPU-GPU output differences with 0.05% threshold

## Requirements

- CUDA Toolkit 11.0+
- GPU with compute capability 8.6+ (Ampere architecture)
- Polybench GPU utilities

## Output

Each execution reports:

- GPU execution time for all three kernel variants
- CPU execution time
- Validation results (number of mismatches)
- GFLOPS performance metrics

Benchmark runs generate timestamped log files (`benchmark_results_YYYYMMDD_HHMMSS.txt`) with detailed statistics.
