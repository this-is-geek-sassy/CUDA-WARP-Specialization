# 3D Convolution FP32

This directory contains various implementations of 3D convolution with different optimization strategies for performance comparison.

## Available Implementations

1. **conv3d_baseline** - Baseline FP32 (no optimization, 32×32 threads)
2. **conv3d_double_buffer** - Double buffering only (no warp specialization)
3. **conv3d_warp_spec** - Double buffering + warp specialization

## Building

### Compile All Implementations

```bash
make
```

This will compile all three implementations and display available executables.

### Compile Individual Implementation

```bash
make conv3d_baseline        # Baseline only
make conv3d_double_buffer   # Double buffering only
make conv3d_warp_spec       # Warp specialization
```

### Clean Build Artifacts

```bash
make clean
```

## Running

Each executable runs the 3D convolution with default parameters:

```bash
./conv3d_baseline        # Run baseline (32×32 threads)
./conv3d_double_buffer   # Run double buffering
./conv3d_warp_spec       # Run warp specialization + double buffering
```

## Benchmarking

### Run Automated Benchmark (All Implementations)

```bash
make benchmark
```

This will:
- Compile all implementations
- Run each version 5 times
- Calculate average execution times
- Display results in a formatted table
- Save results to a timestamped file

### Run Tests (Verification + Timing)

```bash
make test
```

This will compile and run all implementations, displaying both correctness and timing information.

### Manual Benchmark

You can manually benchmark by running each executable multiple times:

```bash
for i in {1..5}; do ./conv3d_baseline; done
```

## Thread Configuration Testing

Benchmark results are available for different thread configurations:

- `16_16.txt` - 16×16 thread block results
- `32_16.txt` - 32×16 thread block results
- `32_32.txt` - 32×32 thread block results
- `64_16.txt` - 64×16 thread block results

## Results Analysis

- `benchmark_results.csv` - CSV format with timing data (in microseconds) for all thread configurations
- `analysis.txt` - Performance analysis and insights
- `WHY_SHARED_MEMORY_FAILS.md` - Analysis of why shared memory optimizations don't help

## Performance Comparison

Based on benchmarks (see `benchmark_results.csv`):

| Thread Config | Baseline | Double Buffer | Warp Spec |
|--------------|----------|---------------|-----------|
| 16×16 | 11,646 μs | 16,580 μs | 12,875 μs |
| 32×16 | 11,598 μs | 16,532 μs | 14,520 μs |
| 32×32 | 14,096 μs | 21,585 μs | 21,334 μs |
| 64×16 | 13,537 μs | 18,992 μs | 21,027 μs |

**Key findings:**
- **Baseline with 32×16 threads is fastest** (11,598 μs)
- Double buffering **degrades performance** by 41-53%
- Warp specialization shows mixed results
- Unlike GEMM, 3D convolution doesn't benefit from these memory optimizations
- The memory access pattern in 3D convolution is different and may need different tuning

## Understanding the Results

For detailed analysis of why shared memory optimizations don't improve performance, see:
- `WHY_SHARED_MEMORY_FAILS.md` - Explains memory access patterns
- `analysis.txt` - Benchmark analysis and conclusions

## Requirements

- CUDA Toolkit with compute capability ≥ 8.9 (configured for sm_89)
- NVIDIA GPU with support for warp specialization features

## Notes

- All implementations use `-O3` optimization flag
- Architecture is set to `sm_89` (adjust in Makefile if needed for your GPU)
- Default thread configuration is 32×32 for baseline (1024 threads = 32 warps per block)
- 3D convolution performs stencil operations on a 3D grid
- The performance characteristics differ significantly from matrix multiplication kernels
