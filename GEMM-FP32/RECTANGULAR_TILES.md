# Rectangular Tiles Support

## Overview

The GEMM kernels have been updated to support **rectangular tiles** with independent dimensions for M, N, and K. This provides more flexibility in tuning performance for different matrix shapes and hardware configurations.

## Tile Dimensions

Three separate tile dimensions can now be configured:

- **TILE_M**: Tile height (rows of output matrix C) - Default: 32
- **TILE_N**: Tile width (columns of output matrix C) - Default: 32
- **TILE_K**: Tile depth (reduction dimension along K) - Default: 32

## Configuration

### Compile-Time Configuration

Define tile dimensions at compile time using preprocessor flags:

```bash
# Example: 32x64 output tiles with 16-element reduction
nvcc -DTILE_M=32 -DTILE_N=64 -DTILE_K=16 gemm_fp_32_cudaDMA.cu -o gemm

# Example: Tall narrow tiles
nvcc -DTILE_M=64 -DTILE_N=16 -DTILE_K=32 gemm_fp_32_cudaDMA.cu -o gemm

# Example: Wide shallow tiles
nvcc -DTILE_M=16 -DTILE_N=64 -DTILE_K=32 gemm_fp_32_cudaDMA.cu -o gemm
```

### Default Configuration

If not specified, all tile dimensions default to 32×32×32 (square tiles), maintaining backward compatibility.

## Memory Requirements

Shared memory usage per thread block:

### Single Buffering:

- **Matrix A tile**: `TILE_M × TILE_K × sizeof(float)` bytes
- **Matrix B tile**: `TILE_K × TILE_N × sizeof(float)` bytes
- **Total**: `(TILE_M × TILE_K + TILE_K × TILE_N) × 4` bytes

### Double Buffering:

- **Total**: `2 × (TILE_M × TILE_K + TILE_K × TILE_N) × 4` bytes

**Examples:**

- 32×32×32: Single=8KB, Double=16KB
- 32×64×16: Single=10KB, Double=20KB
- 64×64×32: Single=24KB, Double=48KB

## Performance Considerations

### Tile Shape Selection

Different tile shapes can be optimal for different scenarios:

**Square Tiles (M=N=K)**

- ✅ Balanced memory access
- ✅ Good for square matrices
- ❌ May not be optimal for rectangular matrices

**Tall Tiles (M>N)**

- ✅ Better for matrices where #rows > #cols
- ✅ More reuse of A tile data
- ⚠️ Higher shared memory for A

**Wide Tiles (N>M)**

- ✅ Better for matrices where #cols > #rows
- ✅ More reuse of B tile data
- ⚠️ Higher shared memory for B

**Small K Dimension**

- ✅ Reduces shared memory usage
- ✅ Allows higher occupancy
- ❌ More tile iterations (more synchronization)

**Large K Dimension**

- ✅ Fewer tile iterations
- ✅ More work per synchronization
- ❌ Higher shared memory usage (lower occupancy)

### Thread Configuration

The number of compute threads must divide the output tile size evenly:

```
TILE_M × TILE_N must be divisible by COMPUTE_THREADS_PER_CTA (256)
```

**Valid combinations:**

- 32×32 = 1024 (1024/256 = 4 elements/thread) ✓
- 32×64 = 2048 (2048/256 = 8 elements/thread) ✓
- 64×64 = 4096 (4096/256 = 16 elements/thread) ✓
- 16×32 = 512 (512/256 = 2 elements/thread) ✓

**Invalid combinations:**

- 30×30 = 900 (900/256 = 3.52 elements/thread) ✗
- 17×17 = 289 (289/256 = 1.13 elements/thread) ✗

### cudaDMA Configuration

The cudaDMA parameters are automatically adjusted based on tile dimensions:

**For Matrix A:**

```cpp
cudaDMAStrided<true, 16, TILE_K * sizeof(fp32_t), DMA_THREADS_PER_LD, TILE_M>
```

- Loads TILE_M rows
- Each row is TILE_K elements
- 32 DMA threads share the work

**For Matrix B:**

```cpp
cudaDMAStrided<true, 16, TILE_N * sizeof(fp32_t), DMA_THREADS_PER_LD, TILE_K>
```

- Loads TILE_K rows
- Each row is TILE_N elements
- 32 DMA threads share the work

## Example Configurations

### Configuration 1: Default Square Tiles

```bash
# No flags needed
make -f Makefile_dma
```

- Tiles: 32×32×32
- Shared Memory: 16KB (double buffer)
- Elements per thread: 4

### Configuration 2: Wide Tiles for Tall Matrices

```bash
nvcc -DTILE_M=64 -DTILE_N=32 -DTILE_K=32 gemm_fp_32_cudaDMA.cu -o gemm
```

- Tiles: 64×32×32
- Shared Memory: 24KB (double buffer)
- Elements per thread: 8
- Best for: Matrices where #rows >> #cols

### Configuration 3: Memory-Efficient

```bash
nvcc -DTILE_M=32 -DTILE_N=32 -DTILE_K=16 gemm_fp_32_cudaDMA.cu -o gemm
```

- Tiles: 32×32×16
- Shared Memory: 10KB (double buffer)
- Elements per thread: 4
- Best for: Limited shared memory, higher occupancy

### Configuration 4: Compute-Heavy

```bash
nvcc -DTILE_M=64 -DTILE_N=64 -DTILE_K=32 gemm_fp_32_cudaDMA.cu -o gemm
```

- Tiles: 64×64×32
- Shared Memory: 48KB (double buffer)
- Elements per thread: 16
- Best for: Large matrices, GPUs with ample shared memory

## Implementation Details

### Changes to Code

1. **Header File (`gemm_fp32_cudaDMA.cuh`)**:

   - Added `TILE_M`, `TILE_N`, `TILE_K` macros
   - Kept `TILE_SIZE` for backward compatibility (equals `TILE_M`)

2. **Kernel Code (`gemm_fp_32_cudaDMA.cu`)**:

   - Updated all three kernels (baseline, single-buffer, double-buffer)
   - Replaced `TILE_SIZE` with appropriate `TILE_M`, `TILE_N`, or `TILE_K`
   - Updated cudaDMA template parameters to use calculated byte counts
   - Updated grid/block dimensions to use `TILE_M` and `TILE_N`

3. **Compute Thread Logic**:
   - Changed fixed array size to `constexpr int elements_per_thread`
   - Dynamically calculated based on `TILE_M × TILE_N`

### Backward Compatibility

All existing code continues to work without modification:

- Default values maintain 32×32×32 tiles
- `TILE_SIZE` macro still available for legacy code
- No changes needed to Makefiles or build scripts

## Tuning Guide

### Step 1: Determine Matrix Shape

- Square matrices: Start with square tiles
- Tall matrices (M >> N): Try increasing TILE_M
- Wide matrices (N >> M): Try increasing TILE_N

### Step 2: Check Shared Memory Limits

```bash
# Query your GPU's shared memory per block
nvidia-smi --query-gpu=compute_cap --format=csv
```

- Compute capability 7.x: 96KB max per block
- Compute capability 8.x: 164KB max per block

### Step 3: Benchmark Different Configurations

```bash
# Test various configurations
for M in 16 32 64; do
  for N in 16 32 64; do
    for K in 16 32 64; do
      nvcc -DTILE_M=$M -DTILE_N=$N -DTILE_K=$K gemm_fp_32_cudaDMA.cu -o gemm_${M}x${N}x${K}
      ./gemm_${M}x${N}x${K}
    done
  done
done
```

### Step 4: Analyze Results

- Look for best execution time
- Check occupancy with `nvprof` or `Nsight Compute`
- Balance between shared memory usage and work per block

## Limitations

1. **TILE_M × TILE_N** must be divisible by 256 (compute thread count)
2. Total shared memory must fit within GPU limits
3. Very small tiles increase synchronization overhead
4. Very large tiles reduce occupancy

## Future Work

- [ ] Auto-tuning script to find optimal tile sizes
- [ ] Support for non-power-of-2 tile dimensions
- [ ] Dynamic tile size selection based on runtime matrix dimensions
- [ ] Integration with cuBLAS-style tile size heuristics

---

**Last Updated:** November 26, 2025
