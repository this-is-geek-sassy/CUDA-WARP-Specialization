# cudaDMA vs cudaDMAv2 API Differences

## Critical API Change: Method Visibility

### The Problem

When migrating from `cudaDMA.h` to `cudaDMAv2.h`, the following compile error occurs:

```
error: function "CudaDMA::wait_for_dma_start" is inaccessible
error: function "CudaDMA::finish_async_dma" is inaccessible
```

---

## Root Cause

### cudaDMA.h (v1) - Public Methods
In the original cudaDMA, synchronization methods are **public**:

```cpp
class CUDADMA_BASE {
public:
    void wait_for_dma_start();   // PUBLIC
    void finish_async_dma();      // PUBLIC
};
```

### cudaDMAv2.h (v2) - Protected Methods
In cudaDMAv2, these methods are **protected**:

```cpp
class CudaDMA {
public:
    void start_async_dma() const;        // PUBLIC (for compute threads)
    void wait_for_dma_finish() const;    // PUBLIC (for compute threads)
    bool owns_this_thread() const;       // PUBLIC
    
protected:
    void wait_for_dma_start() const;     // PROTECTED (internal use only)
    void finish_async_dma() const;       // PROTECTED (internal use only)
};
```

---

## Why This Change?

### Design Philosophy

**cudaDMAv2 enforces better encapsulation:**

1. **`wait_for_dma_start()` and `finish_async_dma()`** are meant to be called **internally** by the `execute_dma()` method
2. User code should **never call these directly** - they are implementation details
3. Always use `execute_dma()` which handles synchronization automatically

---

## API Comparison Table

| Method | cudaDMA.h | cudaDMAv2.h | Purpose | Caller |
|--------|-----------|-------------|---------|--------|
| `start_async_dma()` | Public | Public | Signal DMA to start | **Compute threads** |
| `wait_for_dma_finish()` | Public | Public | Wait for DMA completion | **Compute threads** |
| `execute_dma(src, dst)` | Public | Public | Perform DMA transfer | **DMA threads** |
| `wait_for_dma_start()` | Public | **Protected** | Wait for compute signal | **Internal only** |
| `finish_async_dma()` | Public | **Protected** | Signal DMA completion | **Internal only** |
| `owns_this_thread()` | Public | Public | Check if DMA thread | **All threads** |

---

## Migration Pattern

### ❌ WRONG (cudaDMA.h style - won't compile in v2)

```cpp
// DMA threads for A
else if (dma_ld_a.owns_this_thread())
{
    for (int t = 0; t < numTiles; t++)
    {
        if (in_bounds) {
            fp32_t *src_ptr = &a[...];
            dma_ld_a.execute_dma(src_ptr, As);
        } else {
            // ❌ ERROR: These are protected in cudaDMAv2!
            dma_ld_a.wait_for_dma_start();
            dma_ld_a.finish_async_dma();
        }
    }
}
```

### ✅ CORRECT (cudaDMAv2 style)

```cpp
// DMA threads for A
else if (dma_ld_a.owns_this_thread())
{
    for (int t = 0; t < numTiles; t++)
    {
        // ✅ ALWAYS call execute_dma()
        // It handles synchronization internally
        fp32_t *src_ptr = &a[...];
        dma_ld_a.execute_dma(src_ptr, As);
    }
}
```

---

## Why Always Call execute_dma()?

### Synchronization is Critical

Even when out of bounds, **DMA threads MUST participate in synchronization** to prevent deadlock:

```
Iteration t:
  Compute threads: start_async_dma() → barrier_empty
  DMA threads:     wait_for_dma_start() → barrier_empty ← MUST HAPPEN!
  
  [DMA transfer happens]
  
  DMA threads:     finish_async_dma() → barrier_full ← MUST HAPPEN!
  Compute threads: wait_for_dma_finish() → barrier_full
```

If DMA threads skip synchronization (don't call `execute_dma()`), **compute threads will deadlock** waiting at the barrier.

---

## What About Out-of-Bounds Data?

### It's Safe to Transfer Garbage

```cpp
// Even if out of bounds, this is safe:
fp32_t *src_ptr = &a[aRow * nk + aCol];  // May point to invalid data
dma_ld_a.execute_dma(src_ptr, As);       // Transfers happen, but...
```

**Why it's okay:**
1. Compute threads have boundary checks
2. They only use valid portions of shared memory
3. Invalid data in shared memory is simply ignored
4. Synchronization is maintained correctly

---

## Complete Example: Single Buffering

### cudaDMAv2 Implementation

```cpp
__global__ void gemm_kernel_fp32_cudaDMAv2_single_buffering(...)
{
    __shared__ fp32_t As[TILE_SIZE][TILE_SIZE];
    __shared__ fp32_t Bs[TILE_SIZE][TILE_SIZE];
    
    // Initialize DMA objects
    CudaDMAStrided<true, 16, 128, 128, 32, 32>
        dma_ld_a(0, 256, 256, nk * sizeof(fp32_t), 32 * sizeof(fp32_t));
    
    CudaDMAStrided<true, 16, 128, 128, 32, 32>
        dma_ld_b(1, 256, 288, nj * sizeof(fp32_t), 32 * sizeof(fp32_t));
    
    // Compute threads
    if (threadIdx.x < 256)
    {
        for (int t = 0; t < numTiles; t++) 
        {
            // Signal DMA to start
            dma_ld_a.start_async_dma();
            dma_ld_b.start_async_dma();
            
            // Wait for DMA completion
            dma_ld_a.wait_for_dma_finish();
            dma_ld_b.wait_for_dma_finish();
            
            // Compute using shared memory
            // ...
        }
    }
    // DMA threads for A
    else if (dma_ld_a.owns_this_thread())
    {
        for (int t = 0; t < numTiles; t++)
        {
            // ALWAYS call execute_dma()
            fp32_t *src_ptr = &a[by * TILE_SIZE * nk + t * TILE_SIZE];
            dma_ld_a.execute_dma(src_ptr, As);
        }
    }
    // DMA threads for B
    else if (dma_ld_b.owns_this_thread())
    {
        for (int t = 0; t < numTiles; t++)
        {
            // ALWAYS call execute_dma()
            fp32_t *src_ptr = &b[t * TILE_SIZE * nj + bx * TILE_SIZE];
            dma_ld_b.execute_dma(src_ptr, Bs);
        }
    }
}
```

---

## Internal Implementation (How execute_dma Works)

### Inside execute_dma() Method

```cpp
template<...>
class CudaDMAStrided : public CudaDMA {
public:
    __device__ __forceinline__ 
    void execute_dma(const void *src, void *dst) const
    {
        // Step 1: Wait for compute threads to signal start
        CudaDMA::wait_for_dma_start();  // Protected method
        
        // Step 2: Perform memory transfer
        // ... actual DMA transfer code ...
        
        // Step 3: Signal compute threads that transfer is complete
        CudaDMA::finish_async_dma();    // Protected method
    }
};
```

**This is why we can't call `wait_for_dma_start()` directly** - it's encapsulated!

---

## Double Buffering Pattern

### Correct cudaDMAv2 Implementation

```cpp
// DMA threads for A with double buffering
else if (dma_ld_a.owns_this_thread())
{
    for (int t = 0; t < numTiles; t++)
    {
        int buf_idx = t & 1;  // Ping-pong: 0 or 1
        
        fp32_t *src_ptr = &a[by * TILE_SIZE * nk + t * TILE_SIZE];
        
        // Load into alternating buffer
        if (buf_idx == 0) {
            dma_ld_a.execute_dma(src_ptr, As_0);
        } else {
            dma_ld_a.execute_dma(src_ptr, As_1);
        }
    }
}
```

---

## Summary of Key Changes

### Migration Checklist

1. ✅ Change header: `#include "cudaDMA.h"` → `#include "cudaDMAv2.h"`
2. ✅ Change class name: `cudaDMAStrided` → `CudaDMAStrided` (capital C)
3. ✅ Add `BYTES_PER_THREAD` template parameter (3rd parameter)
4. ✅ **Remove all boundary checks in DMA thread code**
5. ✅ **Always call `execute_dma()` unconditionally**
6. ✅ Never call `wait_for_dma_start()` or `finish_async_dma()` directly

### What Stays the Same

- ✅ `start_async_dma()` and `wait_for_dma_finish()` usage in compute threads
- ✅ `owns_this_thread()` for thread role checking
- ✅ Constructor signature and parameters
- ✅ Overall synchronization pattern

---

## Debugging Tip

### If you see this error:
```
error: function "CudaDMA::wait_for_dma_start" is inaccessible
error: function "CudaDMA::finish_async_dma" is inaccessible
```

### Solution:
Search your code for **direct calls** to these methods and replace with `execute_dma()`:

```bash
# Find problematic code
grep -n "wait_for_dma_start\|finish_async_dma" your_kernel.cu

# Pattern to look for:
# if (condition) {
#     dma.execute_dma(...);
# } else {
#     dma.wait_for_dma_start();   ← DELETE THIS
#     dma.finish_async_dma();      ← DELETE THIS
# }

# Replace with:
# dma.execute_dma(...);  // Always call, no conditions
```

---

## Performance Impact

### No Performance Loss

Removing boundary checks in DMA thread code **does not hurt performance**:

1. ✅ Out-of-bounds transfers are coalesced memory operations (efficient)
2. ✅ Compute threads still have boundary checks (correctness maintained)
3. ✅ Simplifies DMA thread code (fewer branches)
4. ✅ Better encapsulation and maintainability

---

## Conclusion

The key difference in cudaDMAv2 is **enforced encapsulation**:

- **Old way (cudaDMA.h)**: Manual synchronization exposed to user
- **New way (cudaDMAv2.h)**: Synchronization encapsulated in `execute_dma()`

This makes the API **safer and easier to use correctly**, preventing synchronization bugs!
