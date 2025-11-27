# 2D Jacobi Iterative Solver

This directory contains various implementations of the 2D Jacobi iterative method with different optimization strategies for performance comparison.

## Available Implementations

1. **jacobi2d_baseline** - Baseline FP32 (no optimization, 32×16 threads)
2. **jacobi2d_double_buffer** - Double buffering with 2× shared memory tiles
3. **jacobi2d_warp_spec** - Warp specialization + double buffering

## Building

### Compile All Implementations

```bash
make
```

This will compile all three implementations and display available executables.

### Compile Individual Implementation

```bash
make jacobi2d_baseline        # Baseline only
make jacobi2d_double_buffer   # Double buffering only
make jacobi2d_warp_spec       # Warp specialization
```

### Clean Build Artifacts

```bash
make clean
```

## Running

Each executable runs the 2D Jacobi solver with default parameters:

```bash
./jacobi2d_baseline        # Run baseline (32×16 threads)
./jacobi2d_double_buffer   # Run double buffering
./jacobi2d_warp_spec       # Run warp specialization
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
for i in {1..5}; do ./jacobi2d_baseline; done
```

## Thread Configuration Testing

Benchmark results are available for different thread configurations:

- `16_16.txt` - 16×16 thread block results
- `32_16.txt` - 32×16 thread block results  
- `32_32.txt` - 32×32 thread block results
- `64_16.txt` - 64×16 thread block results

## Results Analysis

- `benchmark_results.csv` - CSV format with timing data (in microseconds) for all thread configurations
- `WARP_SPEC_EXPLANATION.md` - Detailed explanation of warp specialization approach

## Performance Comparison

Based on benchmarks (see `benchmark_results.csv`):

| Thread Config | Baseline | Double Buffer | Warp Spec |
|--------------|----------|---------------|-----------|
| 16×16 | 88,460 μs | 88,238 μs | 96,136 μs |
| 32×16 | 88,403 μs | 88,031 μs | 89,434 μs |
| 32×32 | 86,953 μs | 89,230 μs | 112,229 μs |
| 64×16 | 85,807 μs | 88,052 μs | 116,647 μs |

**Key findings:**
- Baseline with 64×16 threads provides best performance
- Double buffering shows marginal improvement in some configurations
- Warp specialization needs tuning for this stencil pattern

## Requirements

- CUDA Toolkit with compute capability ≥ 8.9 (configured for sm_89)
- NVIDIA GPU with support for warp specialization features

## Notes

- All implementations use `-O3` optimization flag
- Architecture is set to `sm_89` (adjust in Makefile if needed for your GPU)
- Default thread configuration is 32×16 for baseline
- The solver performs iterative updates on a 2D grid with stencil operations
