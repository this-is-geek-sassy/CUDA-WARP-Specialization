# CudaDMAStrided Class Hierarchy Documentation (cudaDMAv2.h)

## Overview

The `cudaDMAv2.h` file contains **26 different specializations** of the `CudaDMAStrided` template class. These specializations provide optimized memory transfer implementations for different scenarios in warp-specialized DMA operations.

---

## Class Count Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Primary Template** | 1 | Base template with 6 parameters |
| **Empty Specialization** | 1 | All parameters = 0 (disabled DMA) |
| **1-Param Specializations** | 6 | `ALIGNMENT` + `BYTES_PER_THREAD` |
| **2-Param Specializations** | 9 | + `BYTES_PER_ELMT` |
| **3-Param Specializations** | 9 | + `DMA_THREADS` |
| **Total** | **26** | Full class hierarchy |

---

## Template Parameter Explanation

```cpp
template<bool DO_SYNC, int ALIGNMENT, int BYTES_PER_THREAD, 
         int BYTES_PER_ELMT, int DMA_THREADS, int NUM_ELMTS>
class CudaDMAStrided;
```

### Parameters:
1. **`DO_SYNC`**: Enables/disables explicit synchronization between compute and DMA warps
2. **`ALIGNMENT`**: Memory alignment in bytes (typically 4, 8, or 16)
3. **`BYTES_PER_THREAD`**: Data transferred per DMA thread
4. **`BYTES_PER_ELMT`**: Size of each logical element (e.g., one row of a tile)
5. **`DMA_THREADS`**: Total number of DMA threads
6. **`NUM_ELMTS`**: Number of elements to transfer

---

## Class Specializations by Category

### 1. **Primary Template** (Line 3195)
```cpp
template<bool DO_SYNC, int ALIGNMENT, int BYTES_PER_THREAD, 
         int BYTES_PER_ELMT, int DMA_THREADS, int NUM_ELMTS>
class CudaDMAStrided : public CudaDMA
```
- **Purpose**: Base template (rarely instantiated directly)
- **Usage**: Catches non-specialized cases

---

### 2. **Empty Specialization** (Line 3209)
```cpp
template<bool DO_SYNC>
class CudaDMAStrided<DO_SYNC,0,0,0,0,0>
```
- **Purpose**: Disables DMA functionality
- **Usage**: Compile-time no-op when DMA is not needed
- **Note**: Does NOT inherit from `CudaDMA`

---

### 3. **One-Parameter Specializations** (6 variants)

#### 3a. **Warp-Specialized** (`DO_SYNC=true`) - Lines 5108, 5248, 5394
```cpp
template<int ALIGNMENT, int BYTES_PER_THREAD>
class CudaDMAStrided<true,ALIGNMENT,BYTES_PER_THREAD,0,0,0>
```
- **Purpose**: Runtime-determined element size, warp specialization enabled
- **Synchronization**: Explicit `wait_for_dma_start()` and `finish_async_dma()`
- **Use Case**: When DMA warps and compute warps need explicit handshaking
- **Example**: My GEMM kernel with separate DMA threads

#### 3b. **Non-Warp-Specialized** (`DO_SYNC=false`) - Lines 5553, 5689, 5831
```cpp
template<int ALIGNMENT, int BYTES_PER_THREAD>
class CudaDMAStrided<false,ALIGNMENT,BYTES_PER_THREAD,0,0,0>
```
- **Purpose**: Runtime-determined element size, NO explicit warp synchronization
- **Synchronization**: Implicit via `__syncthreads()` only
- **Use Case**: When ALL threads participate in DMA (no specialization)
- **Example**: Traditional tiled matrix multiplication without DMA warps

---

### 4. **Two-Parameter Specializations** (9 variants)

#### 4a. **Warp-Specialized** (`DO_SYNC=true`) - Lines 6298, 6433, 6574
```cpp
template<int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT>
class CudaDMAStrided<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,0,0>
```
- **Purpose**: Compile-time element size, warp specialization
- **Optimization**: Element size known at compile time → better optimization
- **Use Case**: Fixed-size transfers with dedicated DMA warps

#### 4b. **Non-Warp-Specialized** (`DO_SYNC=false`) - Lines 6728, 6859, 6996
```cpp
template<int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT>
class CudaDMAStrided<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,0,0>
```
- **Purpose**: Compile-time element size, NO warp specialization
- **Use Case**: All threads doing DMA with known element sizes

---

### 5. **Three-Parameter Specializations** (9 variants)

#### 5a. **Warp-Specialized** (`DO_SYNC=true`) - Lines 7480, 7604, 7734
```cpp
template<int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMAStrided<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS,0>
```
- **Purpose**: Fixed DMA thread count, warp specialization
- **Optimization**: Thread count known at compile time
- **Use Case**: My current GEMM implementation! (32 DMA threads)

#### 5b. **Non-Warp-Specialized** (`DO_SYNC=false`) - Lines 7877, 7998, 8125
```cpp
template<int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMAStrided<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS,0>
```
- **Purpose**: Fixed thread count, NO warp specialization
- **Use Case**: All threads doing uniform DMA work

---

### 6. **Four-Parameter Specializations** (6 variants)

#### 6a. **Warp-Specialized** (`DO_SYNC=true`) - Lines 8608, 8726, 8851
```cpp
template<int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT, 
         int DMA_THREADS, int NUM_ELMTS>
class CudaDMAStrided<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS,NUM_ELMTS>
```
- **Purpose**: Fully compile-time specialized, maximum optimization
- **Optimization**: Everything known at compile time → loop unrolling, etc.
- **Use Case**: Ultimate performance for fixed configurations

#### 6b. **Non-Warp-Specialized** (`DO_SYNC=false`) - Lines 8989, 9105, 9227
```cpp
template<int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT, 
         int DMA_THREADS, int NUM_ELMTS>
class CudaDMAStrided<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS,NUM_ELMTS>
```
- **Purpose**: Fully specialized, NO warp specialization
- **Use Case**: Fully optimized but without compute/DMA separation

---

## Why "Non-Warp-Specialized" Classes Exist?

### The Paradox Explained

Although cudaDMA is designed for **warp specialization**, the library also supports **non-warp-specialized** modes. Here's why:

### 1. **Backward Compatibility**
- Allows existing CUDA kernels to use cudaDMA's optimized memory transfer logic
- No need to restructure code with separate DMA warps

### 2. **Hybrid Usage**
```cpp
// Scenario 1: All threads load data (no specialization)
if (threadIdx.x < TOTAL_THREADS) {
    dma.execute_dma(src, dst);  // NO explicit sync
}

// Scenario 2: Dedicated DMA warps (warp specialization)
if (is_dma_thread()) {
    dma.execute_dma(src, dst);  // WITH explicit sync
} else {
    // Compute-only threads
}
```

### 3. **Different Synchronization Models**

| Feature | Warp-Specialized (`DO_SYNC=true`) | Non-Warp-Specialized (`DO_SYNC=false`) |
|---------|-----------------------------------|----------------------------------------|
| **Synchronization** | `wait_for_dma_start()` + `finish_async_dma()` | Implicit `__syncthreads()` |
| **Thread Roles** | Separate compute & DMA warps | All threads do both compute & DMA |
| **Use Case** | Overlapping compute/DMA | Sequential load-then-compute |
| **Complexity** | Higher (manual sync) | Lower (automatic sync) |

### 4. **Key Difference in Implementation**

**Warp-Specialized:**
```cpp
wait_xfer_finish(void *dst_ptr) {
    CudaDMA::wait_for_dma_start();      // ← Wait for compute threads
    // ... perform transfer ...
    CudaDMA::finish_async_dma();         // ← Signal compute threads
}
```

**Non-Warp-Specialized:**
```cpp
wait_xfer_finish(void *dst_ptr) {
    // NO wait_for_dma_start()
    // ... perform transfer ...
    // NO finish_async_dma()
    // Relies on implicit __syncthreads() in user code
}
```

---

## Which Class Am I Using?

My GEMM kernel uses:
```cpp
cudaDMAStrided<true, 16, 128, DMA_THREADS_PER_LD, TILE_SIZE>
```

This maps to **Line 7480** (or similar):
```cpp
template<int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMAStrided<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS,0>
```

**Instantiation:**
- `DO_SYNC = true` → **Warp-specialized**
- `ALIGNMENT = 16` bytes (float4 loads)
- `BYTES_PER_THREAD = 128` bytes
- `BYTES_PER_ELMT = 128` bytes (one row)
- `DMA_THREADS = 32` (one warp)
- `NUM_ELMTS = TILE_SIZE = 32` (runtime parameter)

---

## Summary Table

| `DO_SYNC` | Warp Specialization | Explicit Sync | Use Case |
|-----------|---------------------|---------------|----------|
| `true` | ✅ Yes | ✅ Required | Dedicated DMA warps (my GEMM) |
| `false` | ❌ No | ❌ Implicit | All threads participate in DMA |

---

## Recommendation for My Code

I'm correctly using:
- **`DO_SYNC=true`**: Proper warp specialization
- **Explicit synchronization**: `start_async_dma()`, `wait_for_dma_finish()`
- **Separate thread roles**: DMA threads vs compute threads

This is the **optimal configuration** for maximum compute/DMA overlap in GEMM!

---

## File Statistics

- **Total Lines**: ~14,858
- **CudaDMAStrided Classes**: 26
- **Warp-Specialized**: 13
- **Non-Warp-Specialized**: 12
- **Empty Specialization**: 1
