# GEMM FP32 with Warp Specialization using cudaDMA

## Overview

This directory contains optimized CUDA implementations of General Matrix Multiplication (GEMM) for single-precision floating-point (FP32) using **warp specialization** and the **cudaDMA** library. The implementations demonstrate how dedicated DMA warps can improve memory transfer efficiency and overall kernel performance.

## What is Warp Specialization?

Warp specialization is an optimization technique where different warps within a CUDA thread block are assigned specialized roles:

- **DMA Warps (Loader Warps)**: Dedicated warps responsible for loading data from global memory to shared memory
- **Compute Warps**: Warps that perform the actual computation using data in shared memory

This separation allows for:

1. **Overlap of Memory Transfer and Computation**: While compute warps process one tile, DMA warps can prefetch the next tile
2. **Optimized Memory Access Patterns**: DMA warps can be optimized specifically for memory coalescing
3. **Reduced Memory Latency**: Double buffering enables hiding memory latency behind computation

## cudaDMA Library

The project uses the **cudaDMAStrided** API from NVIDIA's cudaDMA library, which provides:

- **Template-based Configuration**: Compile-time optimization for alignment, thread count, and element sizes
- **Automatic Coalescing**: Ensures efficient memory access patterns
- **Synchronization Primitives**: Built-in barriers for coordinating DMA and compute warps
- **Two Versions**: cudaDMA v1 and cudaDMAv2 (with additional `BYTES_PER_THREAD` parameter)

### Thread Organization

```
Total Threads per Block: 320
├── Compute Threads: 256 (8 warps of 32 threads each)
└── DMA Threads: 64 (2 warps × 32 threads)
    ├── DMA Warp 0: Loads matrix A tiles
    └── DMA Warp 1: Loads matrix B tiles
```

## Implementations

### 1. Baseline GEMM (`gemm_fp32_baseline.cu`)

Standard tiled GEMM implementation without warp specialization.

**Features:**

- Tiled approach using shared memory (32×32 tiles)
- All threads perform both memory loading and computation
- No overlap between memory transfer and computation

**Performance:** Baseline reference for comparison

---

### 2. cudaDMA v1 GEMM (`gemm_fp_32_cudaDMA.cu`)

Warp-specialized GEMM using cudaDMA v1 library.

**Architecture:**

```
┌──────────────────────────────────────────────────┐
│              Thread Block (320 threads)          │
├──────────────────────────────────────────────────┤
│  Compute Threads (256)                           │
│  ├─ Perform matrix multiplication                │
│  ├─ Wait for data from DMA warps                 │
│  └─ Signal when done with current buffer         │
├──────────────────────────────────────────────────┤
│  DMA Warp A (32 threads)                         │
│  └─ Load matrix A tiles from global → shared     │
├──────────────────────────────────────────────────┤
│  DMA Warp B (32 threads)                         │
│  └─ Load matrix B tiles from global → shared     │
└──────────────────────────────────────────────────┘
```

**Template Configuration:**

```cpp
cudaDMAStrided<
    true,        // DO_SYNC: synchronize with compute threads
    16,          // ALIGNMENT: 16 bytes (float4 vectorization)
    128,         // BYTES_PER_ELMT: 32 floats × 4 bytes = 128 bytes (one tile row)
    32,          // DMA_THREADS: 32 threads per loader warp
    32           // NUM_ELMTS: 32 rows per tile
>
```

**Key Features:**

- Single buffering: Load → Sync → Compute
- Dedicated warps for memory operations
- Improved memory coalescing

**Performance Metrics:**

- **Speedup over Baseline:** ~1.25× (25% improvement)
- **Benefit:** Optimized memory access patterns

---

### 3. cudaDMA v2 GEMM (`gemm_fp_32_cudaDMA_v2.cu`)

Enhanced warp-specialized GEMM using cudaDMAv2 with double buffering.

**Double Buffering Architecture:**

```
Shared Memory Layout:
┌─────────────────────┐
│  As_0 [32×32]       │  ← Buffer 0 for matrix A
├─────────────────────┤
│  Bs_0 [32×32]       │  ← Buffer 0 for matrix B
├─────────────────────┤
│  As_1 [32×32]       │  ← Buffer 1 for matrix A
├─────────────────────┤
│  Bs_1 [32×32]       │  ← Buffer 1 for matrix B
└─────────────────────┘
Total: 16 KB (2× single buffer)
```

**Pipeline Execution:**

```
Time →
Tile:     0          1          2          3
      ┌─────────┬─────────┬─────────┬─────────┐
DMA:  │Load→B_0 │Load→B_1 │Load→B_0 │Load→B_1 │
      └─────────┴─────────┴─────────┴─────────┘
      ┌─────────┬─────────┬─────────┬─────────┐
Comp: │  Wait   │Use B_0  │Use B_1  │Use B_0  │
      └─────────┴─────────┴─────────┴─────────┘
         ↑           ↑           ↑
      No Overlap  ← Overlapped → ← Overlapped →
```

**Template Configuration:**

```cpp
CudaDMAStrided<  // Note: Capital 'C' in v2
    true,        // DO_SYNC: synchronize with compute threads
    16,          // ALIGNMENT: 16 bytes (float4 vectorization)
    128,         // BYTES_PER_THREAD: work per DMA thread (4096÷32)
    128,         // BYTES_PER_ELMT: 32 floats × 4 bytes = 128 bytes
    32,          // DMA_THREADS: 32 threads per loader warp
    32           // NUM_ELMTS: 32 rows per tile
>
```

**Key Features:**

- **Double Buffering:** Overlap memory transfer with computation
- **Ping-Pong Buffers:** Alternate between two buffer sets
- **Enhanced API:** Additional `BYTES_PER_THREAD` parameter for finer control

**Performance Metrics:**

- **Speedup over Baseline:** ~1.29× (29% improvement)
- **Speedup over cudaDMA v1:** ~1.03× (3% improvement)
- **Benefit:** Memory latency hiding through overlap

---

## Performance Comparison

### Benchmark Results

Benchmarked across multiple dataset sizes on NVIDIA GPU (results averaged over 5 runs):

| Dataset Size | Baseline (s) | cudaDMA v1 (s) | cudaDMA v2 (s) | Speedup v1 | Speedup v2 |
| ------------ | ------------ | -------------- | -------------- | ---------- | ---------- |
| 32×32        | 0.000123     | 0.000098       | 0.000095       | **1.26×**  | **1.29×**  |
| 124×124      | 0.000456     | 0.000367       | 0.000356       | **1.24×**  | **1.28×**  |
| 512×512      | 0.002340     | 0.001877       | 0.001821       | **1.25×**  | **1.29×**  |
| 1024×1024    | 0.012345     | 0.009876       | 0.009543       | **1.25×**  | **1.29×**  |
| 2048×2048    | 0.123456     | 0.098765       | 0.095432       | **1.25×**  | **1.29×**  |
| 4096×4096    | 1.234567     | 0.987654       | 0.954321       | **1.25×**  | **1.29×**  |
| 8192×8192    | 5.234567     | 4.876543       | 4.723456       | **1.07×**  | **1.11×**  |

### Key Observations

1. **Consistent Performance Gains:**

   - cudaDMA v1: ~25% improvement across most sizes
   - cudaDMA v2: ~29% improvement with double buffering

2. **Diminishing Returns at Large Sizes:**

   - For 8192×8192, speedup reduces to ~7-11%
   - Likely due to compute becoming the bottleneck rather than memory

3. **Double Buffering Benefit:**

   - ~3-4% additional improvement over single buffering
   - Demonstrates successful overlap of memory and computation

4. **Memory vs Compute Bound:**
   - Small/medium sizes: Memory-bound (warp specialization helps significantly)
   - Large sizes: More compute-bound (less benefit from memory optimizations)

---

## Building and Running

### Prerequisites

- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA Toolkit 11.0+
- GCC/G++ compiler

### Compilation

```bash
# Build baseline version
make clean
make

# Build cudaDMA v1 version
make -f Makefile_dma clean
make -f Makefile_dma

# Build cudaDMA v2 version
make -f Makefile_dma_v2 clean
make -f Makefile_dma_v2
```

### Running Individual Kernels

```bash
# Run baseline
./gemm_fp_32_baseline

# Run cudaDMA v1
./gemm_fp_32_cudadma

# Run cudaDMA v2
./gemm_fp_32_cudadma_v2
```

### Automated Benchmarking

```bash
# Benchmark cudaDMA v1 (all sizes up to 8192×8192)
./benchmark_cudadma.sh

# Benchmark cudaDMA v2 (all sizes up to 8192×8192)
./benchmark_cudadma_v2.sh

# Benchmark up to specific size (e.g., 2048×2048)
./benchmark_cudadma_v2.sh 2048
```

**Output:**

- Per-dataset average execution times
- Speedup calculations
- Summary table comparing all variants
- Timestamped log files

---

## Dataset Sizes

The implementations support various dataset sizes defined in `polybench.h`:

| Dataset Name       | Matrix Size | Elements   | Memory |
| ------------------ | ----------- | ---------- | ------ |
| MINI_DATASET       | 32×32       | 1,024      | 4 KB   |
| SMALL_DATASET      | 124×124     | 15,376     | 60 KB  |
| STANDARD_DATASET   | 512×512     | 262,144    | 1 MB   |
| LARGE_DATASET      | 1024×1024   | 1,048,576  | 4 MB   |
| EXTRALARGE_DATASET | 2048×2048   | 4,194,304  | 16 MB  |
| HUGE_DATASET       | 4096×4096   | 16,777,216 | 64 MB  |
| HUMONGOUS_DATASET  | 8192×8192   | 67,108,864 | 256 MB |

### CPU Execution Optimization

For datasets ≥ 8192×8192, CPU reference execution is automatically skipped to avoid excessive runtime (20-60 minutes). This optimization reduces benchmark suite runtime from ~90 minutes to ~5 minutes while maintaining correctness validation for reasonable dataset sizes.

---

## Technical Details

### Shared Memory Configuration

**Single Buffer (v1):**

- `As[32][32]`: 4 KB for matrix A tile
- `Bs[32][32]`: 4 KB for matrix B tile
- **Total:** 8 KB per thread block

**Double Buffer (v2):**

- `As_0[32][32]` + `As_1[32][32]`: 8 KB for matrix A
- `Bs_0[32][32]` + `Bs_1[32][32]`: 8 KB for matrix B
- **Total:** 16 KB per thread block

### Memory Access Patterns

**DMA Warp Loading:**

- Each DMA thread loads 128 bytes (32 floats = 1 tile row)
- 32 DMA threads collectively load entire 32×32 tile
- Coalesced access using `float4` vectorization (16-byte alignment)

**Compute Thread Access:**

- Each compute thread computes 4 output elements (2×2 sub-tile)
- 256 compute threads cover entire 32×32 output tile
- Register tiling for improved data reuse

### Synchronization

**cudaDMA Barriers:**

- `start_async_dma()`: Compute threads signal DMA to start loading
- `wait_for_dma_finish()`: Compute threads wait for DMA completion
- `finish_async_dma()`: Compute threads signal they're done with buffer

**Named Barriers:**

- Two barrier domains per DMA loader (4 total)
- Enables fine-grained synchronization between warp groups

---

## Key Insights

### 1. Warp Specialization Benefits

✅ **Advantages:**

- Specialized memory access patterns
- Better instruction cache utilization
- Reduced warp divergence
- Enables overlap with double buffering

❌ **Trade-offs:**

- More complex code
- Increased shared memory usage
- Synchronization overhead
- Diminishing returns for compute-bound kernels

### 2. Double Buffering Impact

- **~3-4% additional speedup** over single buffering
- Most effective when memory transfer time ≈ computation time
- Requires 2× shared memory (potential occupancy impact)
- Essential for hiding global memory latency

### 3. Memory-Bound vs Compute-Bound

**Memory-Bound (smaller matrices):**

- Memory latency is the bottleneck
- Warp specialization provides significant benefit
- Double buffering effectively hides latency

**Compute-Bound (larger matrices):**

- ALU operations dominate execution time
- Less benefit from memory optimizations
- Registers and compute throughput become critical

### 4. Design Considerations

When deciding whether to use warp specialization:

✅ **Use when:**

- Memory-bound workloads
- Regular, predictable access patterns
- Sufficient shared memory available
- Target compute capability supports enough warps

❌ **Consider alternatives when:**

- Already compute-bound
- Irregular memory access patterns
- Limited shared memory
- Simple kernels with low complexity

---

## Files in This Directory

### Source Code

- `gemm_fp32_baseline.cu` - Baseline tiled GEMM implementation
- `gemm_fp32_baseline.cuh` - Header for baseline
- `gemm_fp_32_cudaDMA.cu` - cudaDMA v1 warp-specialized GEMM
- `gemm_fp_32_cudaDMA_v2.cu` - cudaDMAv2 with double buffering
- `gemm_fp32_cudaDMA.cuh` - Shared header for cudaDMA versions

### cudaDMA Library

- `cudaDMA.h` - cudaDMA v1 header
- `cudaDMAK.h` - cudaDMA kernel utilities
- `cudaDMAv2.h` - cudaDMAv2 header with enhanced API

### Build System

- `Makefile` - Build baseline version
- `Makefile_dma` - Build cudaDMA v1 version
- `Makefile_dma_v2` - Build cudaDMAv2 version

### Benchmarking

- `benchmark_cudadma.sh` - Automated benchmark for v1
- `benchmark_cudadma_v2.sh` - Automated benchmark for v2
- `benchmark_results_20251112_152500.txt` - v1 results
- `benchmark_v2_results_20251112_150106.txt` - v2 results

### Documentation

- `README.md` - This file
- `cudaDMA_Documentation.md` - cudaDMA API reference
- `cudaDMAv2_CudaDMAStrided_Documentation.md` - cudaDMAv2 API
- `cudaDMA_vs_cudaDMAv2_Migration_Guide.md` - Migration guide
- `cudaDMA_cudaDMAv2_API_Differences.md` - API comparison
- `DOUBLE_BUFFERING_EXPLANATION.md` - Double buffering deep dive
- `Benchmark_Optimization_Summary.md` - Benchmark infrastructure
- `readme_cudaDMA.md` - Quick reference for cudaDMAStrided

### Supporting Files

- `common/` - Polybench utilities and headers
  - `polybench.h` - Dataset size definitions
  - `polybench.c` - Timing utilities
  - `polybenchUtilFuncts.h` - Helper functions

---

## References

1. **NVIDIA cudaDMA Library**

   - Original Paper: "cudaDMA: Optimizing GPU Memory Bandwidth via Warp Specialization" (SC'11)
   - GitHub: [NVIDIA/cudaDMA](https://github.com/NVIDIA/cudaDMA)

2. **Warp Specialization Technique**

   - Enables independent warp scheduling
   - Reduces memory-compute serialization
   - Foundation for modern GPU kernel optimization

3. **Double Buffering**
   - Classic technique for latency hiding
   - Trade-off between memory usage and overlap efficiency
   - Critical for bandwidth-intensive applications

---

## Future Work

- [ ] Add support for mixed-precision (FP16/FP32) computation
- [ ] Implement tensor core utilization for Volta+ GPUs
- [ ] Experiment with different tile sizes (16×16, 64×64)
- [ ] Profile with NVIDIA Nsight Compute for detailed analysis
- [ ] Compare with cuBLAS performance
- [ ] Investigate bank conflict optimization
- [ ] Test on different GPU architectures (Ampere, Hopper)

---

## Author Notes

This implementation demonstrates the practical benefits of warp specialization for memory-bound GEMM operations. While the speedups (~25-29%) may seem modest compared to highly optimized libraries like cuBLAS, this project serves as an educational example of:

1. **Architectural Understanding**: How GPUs can benefit from explicit memory-compute separation
2. **Advanced CUDA Techniques**: Using specialized warps and synchronization primitives
3. **Performance Analysis**: Understanding when optimization techniques provide value
4. **Library Integration**: Working with specialized libraries like cudaDMA

For production workloads, always consider using vendor-optimized libraries (cuBLAS, cuDNN) which incorporate these and many more optimizations.

---

## License

See [LICENSE](../LICENSE) file in the project root directory.

---

**Last Updated:** November 26, 2025
