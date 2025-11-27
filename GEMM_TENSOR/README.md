# GEMM Tensor Core Implementations

This directory contains various GEMM (General Matrix Multiply) implementations using NVIDIA Tensor Cores with different optimization strategies.

## Available Implementations

1. **gemm_tensor** - Baseline implementation (FP16 with FP32 accumulation)
2. **gemm_tensor_double_buffer** - Double buffering without warp specialization
3. **gemm_warp_spec_single_cpasync** - Warp specialization + single buffer + cp.async
4. **gemm_warp_spec_double_cpasync** - Warp specialization + double buffering + cp.async
5. **gemm_tf32** - TF32 precision (FP32 input with TF32 tensor cores)

## Building

### Compile All Implementations

```bash
make
```

This will compile all five implementations and display available executables.

### Compile Individual Implementation

```bash
make gemm_tensor                      # Baseline only
make gemm_tensor_double_buffer        # Double buffering only
make gemm_warp_spec_single_cpasync    # Warp spec + single buffer
make gemm_warp_spec_double_cpasync    # Warp spec + double buffer
make gemm_tf32                        # TF32 version
```

### Clean Build Artifacts

```bash
make clean
```

## Running

Each executable runs the GEMM operation with matrix size 2048×2048:

```bash
./gemm_tensor                      # Run baseline
./gemm_tensor_double_buffer        # Run double buffering
./gemm_warp_spec_single_cpasync    # Run warp spec + single buffer
./gemm_warp_spec_double_cpasync    # Run warp spec + double buffer
./gemm_tf32                        # Run TF32 version
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
- Save results to a timestamped file (e.g., `128.txt`, `64.txt`, etc.)

### Manual Benchmark

You can manually benchmark by running each executable multiple times:

```bash
for i in {1..5}; do ./gemm_tensor; done
```

## Results Analysis

Benchmark results are saved in:
- `benchmark_results.csv` - CSV format with timing data (in microseconds)
- `16.txt`, `32.txt`, `64.txt`, `128.txt` - Raw benchmark outputs for different block sizes
- `RECTANGULAR_TILES_RESULTS.md` - Analysis of rectangular tile configurations

## Performance Tips

- **Best for small matrices**: Baseline or double buffering
- **Best for large matrices**: Warp specialization with double buffering + cp.async
- **TF32 version**: Use when FP32 precision is required with tensor core acceleration

## Requirements

- CUDA Toolkit with compute capability ≥ 8.9 (configured for sm_89)
- NVIDIA GPU with Tensor Core support (Ampere, Ada Lovelace, or Hopper architecture)
- For TF32: Ampere or newer architecture

## Notes

- All implementations use `-O3` optimization flag
- Architecture is set to `sm_89` (adjust in Makefile if needed for your GPU)
- Tensor Core operations require proper alignment and data types (FP16/TF32)
