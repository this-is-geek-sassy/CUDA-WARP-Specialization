# cudaDMA vs cudaDMAv2 Migration Guide

## Overview

This guide explains the differences between cudaDMA.h (v1) and cudaDMAv2.h, and how to migrate your GEMM kernel.

---

## Template Parameter Comparison

### cudaDMA.h (v1)
```cpp
template<bool DO_SYNC, int ALIGNMENT, int BYTES_PER_ELMT, 
         int DMA_THREADS, int NUM_ELMTS>
class cudaDMAStrided;
```

### cudaDMAv2.h (v2)
```cpp
template<bool DO_SYNC, int ALIGNMENT, int BYTES_PER_THREAD,
         int BYTES_PER_ELMT, int DMA_THREADS, int NUM_ELMTS>
class CudaDMAStrided;  // Note: capital 'C'
```

### Key Difference
**cudaDMAv2 adds `BYTES_PER_THREAD` parameter!**

---

## Parameter Explanation for My GEMM

### My Configuration:
- **Tile Size**: 32×32 elements
- **Element Type**: `float` (4 bytes)
- **DMA Threads**: 32 (one warp)

### Calculation:

| Parameter | Formula | Value | Explanation |
|-----------|---------|-------|-------------|
| **Total Tile Bytes** | 32 × 32 × 4 | 4096 bytes | One full tile |
| **DMA Threads** | - | 32 threads | One warp |
| **BYTES_PER_THREAD** | 4096 ÷ 32 | **128 bytes** | Each DMA thread's workload |
| **BYTES_PER_ELMT** | 32 × 4 | **128 bytes** | One row of tile |
| **NUM_ELMTS** | - | 32 | Number of rows |

### Important Notes:

1. **BYTES_PER_THREAD = BYTES_PER_ELMT** in my case
   - Each DMA thread transfers exactly one row
   - Not always true for other configurations!

2. **BYTES_PER_THREAD** is about **DMA thread workload**, NOT compute thread workload
   - I have 256 compute threads
   - But only 32 DMA threads
   - DMA threads do the memory transfer work

3. **"1 element per thread"** is WRONG interpretation
   - That's for compute threads (256 threads, each processes 4 output elements)
   - For DMA threads: 32 threads, each transfers 128 bytes (32 floats = 1 row)

---

## Migration Example

### Current Code (cudaDMA v1)

```cpp
#include "cudaDMA.h"

cudaDMAStrided<true, 16, 128, DMA_THREADS_PER_LD, TILE_SIZE>
//             ^     ^   ^    ^                   ^
//             |     |   |    |                   NUM_ELMTS (runtime)
//             |     |   |    DMA_THREADS (32)
//             |     |   BYTES_PER_ELMT (128 bytes = 1 row)
//             |     ALIGNMENT (16 bytes = float4)
//             DO_SYNC (warp-specialized)
    dma_ld_a(0, COMPUTE_THREADS_PER_CTA, COMPUTE_THREADS_PER_CTA,
             nk * sizeof(fp32_t), TILE_SIZE * sizeof(fp32_t));
```

### Migrated Code (cudaDMAv2)

```cpp
#include "cudaDMAv2.h"

CudaDMAStrided<true, 16, 128, 128, 32, 32>
//             ^     ^   ^    ^    ^   ^
//             |     |   |    |    |   NUM_ELMTS (32 rows)
//             |     |   |    |    DMA_THREADS (32)
//             |     |   |    BYTES_PER_ELMT (128 bytes)
//             |     |   BYTES_PER_THREAD (128 bytes)
//             |     ALIGNMENT (16 bytes)
//             DO_SYNC (warp-specialized)
    dma_ld_a(0, COMPUTE_THREADS_PER_CTA, COMPUTE_THREADS_PER_CTA,
             nk * sizeof(fp32_t), TILE_SIZE * sizeof(fp32_t));
```

---

## Common Misunderstandings

### ❌ WRONG: "Each thread processes 1 element"
This confuses **compute threads** with **DMA threads**.

**Compute threads** (256):
- Each processes 4 output elements (32×32 ÷ 256 = 4)

**DMA threads** (32):
- Each transfers 128 bytes = 32 floats = 1 row

### ❌ WRONG: "BYTES_PER_THREAD = 4 bytes"
This would mean each DMA thread only transfers 1 float!
- With 32 DMA threads × 4 bytes = 128 bytes total
- But you need to transfer 4096 bytes!

### ✅ CORRECT: "BYTES_PER_THREAD = 128 bytes"
- 32 DMA threads × 128 bytes = 4096 bytes ✓
- Each DMA thread transfers one complete row (32 floats)

---

## Visual Representation

### DMA Thread Workload Distribution

```
Tile: 32×32 floats = 4096 bytes
┌─────────────────────────────────┐
│ Row  0 (128 bytes) → DMA Thread 0
│ Row  1 (128 bytes) → DMA Thread 1
│ Row  2 (128 bytes) → DMA Thread 2
│ ...
│ Row 30 (128 bytes) → DMA Thread 30
│ Row 31 (128 bytes) → DMA Thread 31
└─────────────────────────────────┘

Each DMA thread: BYTES_PER_THREAD = 128 bytes
Total: 32 threads × 128 bytes = 4096 bytes ✓
```

### Compute Thread Workload Distribution

```
Output Tile: 32×32 = 1024 elements
Compute Threads: 256

Each compute thread: 1024 ÷ 256 = 4 elements

Thread 0   → elements [0, 1, 2, 3]
Thread 1   → elements [4, 5, 6, 7]
...
Thread 255 → elements [1020, 1021, 1022, 1023]
```

**These are SEPARATE concerns!**

---

## Full Kernel Migration

### Changes Required:

1. **Header file**: `cudaDMA.h` → `cudaDMAv2.h`
2. **Class name**: `cudaDMAStrided` → `CudaDMAStrided` (capital C)
3. **Template parameters**: Add `BYTES_PER_THREAD` as 3rd parameter
4. **Make all parameters compile-time** (optional but recommended for performance)

### Before (cudaDMA v1):
```cpp
#include "cudaDMA.h"

__global__ void gemm_kernel_fp32_cudaDMA_single_buffering(...)
{
    __shared__ fp32_t As[TILE_SIZE][TILE_SIZE];
    __shared__ fp32_t Bs[TILE_SIZE][TILE_SIZE];
    
    cudaDMAStrided<true, 16, 128, DMA_THREADS_PER_LD, TILE_SIZE>
        dma_ld_a(0, COMPUTE_THREADS_PER_CTA, COMPUTE_THREADS_PER_CTA,
                 nk * sizeof(fp32_t), TILE_SIZE * sizeof(fp32_t));
    
    cudaDMAStrided<true, 16, 128, DMA_THREADS_PER_LD, TILE_SIZE>
        dma_ld_b(1, COMPUTE_THREADS_PER_CTA, 
                 COMPUTE_THREADS_PER_CTA + DMA_THREADS_PER_LD,
                 nj * sizeof(fp32_t), TILE_SIZE * sizeof(fp32_t));
    // ... rest of kernel
}
```

### After (cudaDMAv2):
```cpp
#include "cudaDMAv2.h"

__global__ void gemm_kernel_fp32_cudaDMAv2_single_buffering(...)
{
    __shared__ fp32_t As[TILE_SIZE][TILE_SIZE];
    __shared__ fp32_t Bs[TILE_SIZE][TILE_SIZE];
    
    CudaDMAStrided<true,    // DO_SYNC
                   16,      // ALIGNMENT (float4)
                   128,     // BYTES_PER_THREAD (per DMA thread)
                   128,     // BYTES_PER_ELMT (one row)
                   32,      // DMA_THREADS (one warp)
                   32>      // NUM_ELMTS (32 rows)
        dma_ld_a(0, COMPUTE_THREADS_PER_CTA, COMPUTE_THREADS_PER_CTA,
                 nk * sizeof(fp32_t), TILE_SIZE * sizeof(fp32_t));
    
    CudaDMAStrided<true, 16, 128, 128, 32, 32>
        dma_ld_b(1, COMPUTE_THREADS_PER_CTA, 
                 COMPUTE_THREADS_PER_CTA + DMA_THREADS_PER_LD,
                 nj * sizeof(fp32_t), TILE_SIZE * sizeof(fp32_t));
    // ... rest of kernel (NO CHANGES NEEDED)
}
```

---

## API Compatibility

### Good News: Constructor and Methods are IDENTICAL!

```cpp
// Constructor signature (SAME in both versions)
CudaDMAStrided(const int dmaID,
               const int num_compute_threads,
               const int dma_threadIdx_start,
               const int src_stride,
               const int elmt_stride);

// Methods (SAME in both versions)
bool owns_this_thread();
void start_async_dma();
void wait_for_dma_finish();
void execute_dma(const void *src, void *dst);
void wait_for_dma_start();
void finish_async_dma();
```

**No code changes needed** in kernel body, only in template parameters!

---

## Performance Expectations

### Potential Benefits of cudaDMAv2:

1. **Better Compile-Time Optimization**
   - More template parameters known at compile time
   - Compiler can unroll loops, eliminate branches

2. **Explicit BYTES_PER_THREAD**
   - Clearer thread workload distribution
   - Better memory coalescing opportunities

3. **More Specializations**
   - 26 specializations (vs 9 in v1)
   - Better matched to your specific configuration

### Expected Performance Impact:

| Metric | cudaDMA v1 | cudaDMAv2 | Improvement |
|--------|------------|-----------|-------------|
| Compile-time optimization | Good | Better | +5-10% |
| Memory coalescing | Good | Same | 0% |
| Code complexity | Lower | Same | - |

**Note**: For my specific configuration, improvements may be modest because I'm already using optimal parameters.

---

## Debugging Tips

### 1. Verify Thread Counts
```cpp
// Add at kernel start
if (threadIdx.x == 0) {
    printf("Total threads: %d\n", TOTAL_THREADS);
    printf("Compute threads: %d\n", COMPUTE_THREADS_PER_CTA);
    printf("DMA threads per loader: %d\n", DMA_THREADS_PER_LD);
}
```

### 2. Verify DMA Thread Ownership
```cpp
// In DMA thread section
if (dma_ld_a.owns_this_thread()) {
    if (threadIdx.x == COMPUTE_THREADS_PER_CTA) {
        printf("First DMA-A thread: %d\n", threadIdx.x);
    }
}
```

### 3. Check Memory Transfer Size
```cpp
// Expected: 4096 bytes per tile
int expected_bytes = TILE_SIZE * TILE_SIZE * sizeof(fp32_t);
printf("Expected transfer: %d bytes\n", expected_bytes);
```

---

## Quick Reference Table

| Concept | Value | Explanation |
|---------|-------|-------------|
| **Tile dimensions** | 32×32 | Square tile |
| **Total tile elements** | 1024 | 32 × 32 |
| **Element size** | 4 bytes | sizeof(float) |
| **Total tile bytes** | 4096 bytes | 1024 × 4 |
| **DMA threads** | 32 | One warp |
| **BYTES_PER_THREAD** | **128 bytes** | 4096 ÷ 32 |
| **BYTES_PER_ELMT** | 128 bytes | One row (32 floats) |
| **NUM_ELMTS** | 32 | Number of rows |
| **Compute threads** | 256 | 8 warps |
| **Elements per compute thread** | 4 | 1024 ÷ 256 |

---

## Summary

### The Key Formula:
```
BYTES_PER_THREAD = (Total Tile Bytes) ÷ (Number of DMA Threads)
                 = (32 × 32 × 4) ÷ 32
                 = 4096 ÷ 32
                 = 128 bytes
```

### NOT:
```
❌ BYTES_PER_THREAD ≠ sizeof(element)           = 4 bytes (WRONG!)
❌ BYTES_PER_THREAD ≠ elements per compute thread × 4 = 16 bytes (WRONG!)
✅ BYTES_PER_THREAD = total bytes ÷ DMA threads = 128 bytes (CORRECT!)
```

### Remember:
- **BYTES_PER_THREAD** is about **DMA thread workload**
- Each DMA thread transfers **one complete row** (32 floats = 128 bytes)
- Compute thread workload is a **separate concept**
