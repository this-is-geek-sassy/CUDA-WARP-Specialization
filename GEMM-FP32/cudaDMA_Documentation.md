# cudaDMA.h Class Hierarchy Documentation

## Overview

The `cudaDMA.h` file is the **original/legacy version** of the cudaDMA library, providing three main DMA transfer patterns through template class hierarchies. This file contains **35 different class specializations** across three main DMA classes.

---

## File Comparison: cudaDMA.h vs cudaDMAv2.h

| Feature | cudaDMA.h (Legacy) | cudaDMAv2.h (Version 2) |
|---------|-------------------|------------------------|
| **Release** | Original | Updated/Optimized |
| **Total Classes** | 35 | 26 |
| **DMA Patterns** | 3 (Sequential, Strided, Indirect) | 2 (Strided, Indirect) |
| **Class Hierarchy** | Simpler | More complex with additional optimizations |
| **Use Case** | General purpose | Performance-critical applications |

---

## Class Count Summary

| Class Type | Count | Description |
|------------|-------|-------------|
| **cudaDMASequential** | 7 | Contiguous memory transfers |
| **cudaDMAStrided** | 9 | Regular strided memory access |
| **cudaDMAIndirect** | 18 | Gather/scatter with index array |
| **Base Classes** | 1 | cudaDMAStridedBase, cudaDMAIndirectBase |
| **Total** | **35** | Complete hierarchy |

---

## 1. cudaDMASequential (7 Specializations)

### Purpose
Handles **contiguous memory transfers** where source and destination are sequential in memory.

### Template Signature
```cpp
template<bool DO_SYNC, int ALIGNMENT, int BYTES_PER_ELMT=0, int DMA_THREADS=0>
class cudaDMASequential : public CUDADMA_BASE
```

### Parameters
- **`DO_SYNC`**: Enable warp specialization synchronization
- **`ALIGNMENT`**: Memory alignment (4, 8, or 16 bytes)
- **`BYTES_PER_ELMT`**: Size of each element in bytes
- **`DMA_THREADS`**: Number of DMA threads

### Specializations

#### 1.1 Primary Template (Line 644)
```cpp
template<bool DO_SYNC, int ALIGNMENT, int BYTES_PER_ELMT=0, int DMA_THREADS=0>
class cudaDMASequential : public CUDADMA_BASE
```
- **Purpose**: Fully parameterized sequential transfer
- **Use Case**: General contiguous memory transfers
- **Synchronization**: Controlled by `DO_SYNC`

#### 1.2 Four-Parameter, Non-Warp-Specialized (Line 850)
```cpp
template<int ALIGNMENT, int BYTES_PER_ELMT, int DMA_THREADS>
class cudaDMASequential<false,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
```
- **Purpose**: Compile-time optimized, no warp specialization
- **Use Case**: When all threads participate in DMA

#### 1.3 One-Parameter, Warp-Specialized (Line 906)
```cpp
template<int ALIGNMENT>
class cudaDMASequential<true,ALIGNMENT,0,0>
```
- **Purpose**: Runtime element size, warp specialization enabled
- **Use Case**: Variable-sized sequential transfers with dedicated DMA warps

#### 1.4 One-Parameter, Non-Warp-Specialized (Line 965)
```cpp
template<int ALIGNMENT>
class cudaDMASequential<false,ALIGNMENT,0,0>
```
- **Purpose**: Runtime element size, no warp specialization
- **Use Case**: Simple sequential transfers by all threads

#### 1.5 Two-Parameter, Warp-Specialized (Line 1028)
```cpp
template<int ALIGNMENT, int BYTES_PER_ELMT>
class cudaDMASequential<true,ALIGNMENT,BYTES_PER_ELMT,0>
```
- **Purpose**: Fixed element size, runtime thread count, warp specialization
- **Use Case**: Known element size with dedicated DMA warps

#### 1.6 Two-Parameter, Non-Warp-Specialized (Line 1084)
```cpp
template<int ALIGNMENT, int BYTES_PER_ELMT>
class cudaDMASequential<false,ALIGNMENT,BYTES_PER_ELMT,0>
```
- **Purpose**: Fixed element size, no warp specialization
- **Use Case**: All threads do sequential DMA with known element size

---

## 2. cudaDMAStrided (9 Specializations)

### Purpose
Handles **strided memory access patterns** where elements are at regular intervals (e.g., loading matrix rows).

### Template Signature
```cpp
template<bool DO_SYNC, int ALIGNMENT, int BYTES_PER_ELMT=0, int DMA_THREADS=0, int NUM_ELMTS=0>
class cudaDMAStrided : public cudaDMAStridedBase
```

### Parameters
- **`DO_SYNC`**: Enable warp specialization synchronization
- **`ALIGNMENT`**: Memory alignment (4, 8, or 16 bytes)
- **`BYTES_PER_ELMT`**: Size of each element in bytes
- **`DMA_THREADS`**: Number of DMA threads
- **`NUM_ELMTS`**: Number of elements to transfer

### Specializations

#### 2.1 Primary Template (Line 1977)
```cpp
template<bool DO_SYNC, int ALIGNMENT, int BYTES_PER_ELMT=0, int DMA_THREADS=0, int NUM_ELMTS=0>
class cudaDMAStrided : public cudaDMAStridedBase
```
- **Purpose**: Most general strided transfer pattern
- **Use Case**: **MY GEMM KERNEL USES THIS!**
- **Example**: `cudaDMAStrided<true, 16, 128, 32, 32>`

#### 2.2 Four-Parameter, Non-Warp-Specialized (Line 2163)
```cpp
template<int ALIGNMENT, int BYTES_PER_ELMT, int DMA_THREADS, int NUM_ELMTS>
class cudaDMAStrided<false,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS,NUM_ELMTS>
```
- **Purpose**: Fully compile-time optimized, no warp specialization
- **Use Case**: Maximum optimization when all threads do DMA

#### 2.3 One-Parameter, Warp-Specialized (Line 2228)
```cpp
template<int ALIGNMENT>
class cudaDMAStrided<true,ALIGNMENT,0,0,0>
```
- **Purpose**: Runtime parameters, warp specialization
- **Use Case**: Flexible configuration with dedicated DMA warps

#### 2.4 One-Parameter, Non-Warp-Specialized (Line 2315)
```cpp
template<int ALIGNMENT>
class cudaDMAStrided<false,ALIGNMENT,0,0,0>
```
- **Purpose**: Runtime parameters, no warp specialization
- **Use Case**: All threads participate in strided DMA

#### 2.5 Two-Parameter, Warp-Specialized (Line 2402)
```cpp
template<int ALIGNMENT, int BYTES_PER_ELMT>
class cudaDMAStrided<true,ALIGNMENT,BYTES_PER_ELMT,0,0>
```
- **Purpose**: Fixed element size, warp specialization
- **Use Case**: Known element size with dedicated DMA warps

#### 2.6 Two-Parameter, Non-Warp-Specialized (Line 2480)
```cpp
template<int ALIGNMENT, int BYTES_PER_ELMT>
class cudaDMAStrided<false,ALIGNMENT,BYTES_PER_ELMT,0,0>
```
- **Purpose**: Fixed element size, no warp specialization
- **Use Case**: All threads do strided DMA with known element size

#### 2.7 Three-Parameter, Warp-Specialized (Line 2558)
```cpp
template<int ALIGNMENT, int BYTES_PER_ELMT, int DMA_THREADS>
class cudaDMAStrided<true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS,0>
```
- **Purpose**: Fixed element size and thread count, warp specialization
- **Use Case**: Highly optimized for specific configurations

#### 2.8 Three-Parameter, Non-Warp-Specialized (Line 2634)
```cpp
template<int ALIGNMENT, int BYTES_PER_ELMT, int DMA_THREADS>
class cudaDMAStrided<false,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS,0>
```
- **Purpose**: Fixed element size and thread count, no warp specialization
- **Use Case**: Optimized strided DMA without warp specialization

---

## 3. cudaDMAIndirect (18 Specializations)

### Purpose
Handles **indirect memory access** (gather/scatter) where element locations are specified by an index array.

### Template Signature
```cpp
template<bool GATHER, bool DO_SYNC, int ALIGNMENT, int BYTES_PER_ELMT=0, 
         int DMA_THREADS=0, int NUM_ELMTS=0>
class cudaDMAIndirect : public cudaDMAIndirectBase<GATHER>
```

### Additional Parameter
- **`GATHER`**: 
  - `true` = Gather operation (read from scattered locations → contiguous)
  - `false` = Scatter operation (contiguous → scattered locations)

### Specializations

#### 3.1 Primary Template (Line 2891)
```cpp
template<bool GATHER, bool DO_SYNC, int ALIGNMENT, int BYTES_PER_ELMT=0, 
         int DMA_THREADS=0, int NUM_ELMTS=0>
class cudaDMAIndirect : public cudaDMAIndirectBase<GATHER>
```
- **Purpose**: Fully parameterized gather/scatter
- **Use Case**: Indirect memory access with index array
- **Operations**: Both gather and scatter supported

#### 3.2-3.18 Various Specializations
Similar pattern to `cudaDMAStrided`, but with additional `GATHER` parameter:

**Warp-Specialized Gather/Scatter** (Lines 3157, 3311, 3449):
- Dedicated DMA warps for indirect access
- Explicit synchronization with compute warps

**Non-Warp-Specialized Gather/Scatter** (Lines 3092, 3232, 3378, 3516):
- All threads participate in indirect access
- Implicit synchronization only

---

## Key Differences: Warp-Specialized vs Non-Warp-Specialized

### Synchronization Mechanism

#### Warp-Specialized (`DO_SYNC=true`)
```cpp
// In execute_dma() or STRIDED_EXECUTE macro
if (DO_SYNC) {
    CUDADMA_BASE::wait_for_dma_start();    // Wait for compute threads
    // ... perform DMA transfer ...
    CUDADMA_BASE::finish_async_dma();      // Signal compute threads
}
```

#### Non-Warp-Specialized (`DO_SYNC=false`)
```cpp
// In execute_dma() - simplified
// NO wait_for_dma_start()
// ... perform DMA transfer ...
// NO finish_async_dma()
// Relies on __syncthreads() in user code
```

---

## Memory Access Patterns

### 1. Sequential (cudaDMASequential)
```
Memory: [A][B][C][D][E][F][G][H]
Access: Sequential read/write
```
**Use Case**: Copying contiguous arrays, buffers

### 2. Strided (cudaDMAStrided)
```
Memory: [A]...[B]...[C]...[D]
Access: Regular stride between elements
```
**Use Case**: Matrix rows, array-of-structs field access, **your GEMM kernel**

### 3. Indirect (cudaDMAIndirect)
```
Memory:  [A][B][C][D][E][F][G][H]
Index:   [3, 0, 7, 2, ...]
Access:  Irregular pattern via index array
```
**Use Case**: Sparse matrices, graph algorithms, gather/scatter operations

---

## Macro-Based Execution

The library uses extensive macros for code generation:

### Key Macros

#### SEQUENTIAL_EXECUTE
```cpp
#define SEQUENTIAL_EXECUTE(DO_SYNC,BYTES_PER_ELMT,DMA_THREADS)
```
- Generates code for sequential memory transfers
- Handles partial threads and alignment

#### STRIDED_EXECUTE
```cpp
#define STRIDED_EXECUTE(DO_SYNC)
```
- Generates code for strided memory transfers
- Handles element splitting and row/column iterations

#### INDIRECT_EXECUTE
```cpp
#define INDIRECT_EXECUTE(DO_SYNC)
```
- Generates code for indirect (gather/scatter) transfers
- Uses index array to determine memory locations

---

## Performance Considerations

### 1. Compile-Time vs Runtime Parameters

| Configuration | Optimization Level | Flexibility |
|---------------|-------------------|-------------|
| All compile-time | ⭐⭐⭐⭐⭐ Highest | ⭐ Lowest |
| Mixed | ⭐⭐⭐ Moderate | ⭐⭐⭐ Moderate |
| All runtime | ⭐ Lowest | ⭐⭐⭐⭐⭐ Highest |

### 2. Memory Alignment

```cpp
ALIGNMENT = 4  → float   (4 bytes)
ALIGNMENT = 8  → float2  (8 bytes)  
ALIGNMENT = 16 → float4  (16 bytes) ⭐ Best for modern GPUs
```

### 3. Thread Configuration

```cpp
MAX_BYTES_OUTSTANDING_PER_THREAD = 4 * ALIGNMENT
MAX_LDS_OUTSTANDING_PER_THREAD = 4
```
- Limits outstanding memory operations per thread
- Prevents register pressure

---

## My GEMM Kernel Configuration

```cpp
cudaDMAStrided<true, 16, 128, DMA_THREADS_PER_LD, TILE_SIZE>
    dma_ld_a(0, COMPUTE_THREADS_PER_CTA, COMPUTE_THREADS_PER_CTA,
             nk * sizeof(fp32_t), TILE_SIZE * sizeof(fp32_t));
```

**Maps to**: Primary template at Line 1977

**Configuration**:
- ✅ **Warp-Specialized** (`DO_SYNC=true`)
- ✅ **16-byte aligned** (float4 loads)
- ✅ **128 bytes per thread**
- ✅ **32 DMA threads** (1 warp)
- ✅ **32 elements** (TILE_SIZE, runtime)

**Pattern**: Strided access (loading matrix rows with stride)

---

## Class Hierarchy Visualization

```
CUDADMA_BASE (Base class)
│
├── cudaDMASequential (7 variants)
│   ├── Warp-Specialized (DO_SYNC=true)
│   └── Non-Warp-Specialized (DO_SYNC=false)
│
├── cudaDMAStridedBase (Helper base class)
│   │
│   └── cudaDMAStrided (9 variants)
│       ├── Warp-Specialized (DO_SYNC=true) ← MY GEMM
│       └── Non-Warp-Specialized (DO_SYNC=false)
│
└── cudaDMAIndirectBase<GATHER> (Helper base class)
    │
    └── cudaDMAIndirect (18 variants)
        ├── Gather (GATHER=true)
        │   ├── Warp-Specialized
        │   └── Non-Warp-Specialized
        └── Scatter (GATHER=false)
            ├── Warp-Specialized
            └── Non-Warp-Specialized
```

---

## When to Use Each Class

### Use cudaDMASequential when:
- ✅ Transferring contiguous memory blocks
- ✅ Simple array copies
- ✅ No complex access patterns

### Use cudaDMAStrided when:
- ✅ Regular strided access (matrix rows)
- ✅ Array-of-structs field access
- ✅ **GEMM tiling (my use case!)**

### Use cudaDMAIndirect when:
- ✅ Irregular access patterns
- ✅ Sparse matrix operations
- ✅ Graph algorithms
- ✅ Gather/scatter operations

---

## Common Pitfalls

### 1. Synchronization Mismatch
```cpp
// ❌ WRONG: Using DO_SYNC=false but calling sync functions
cudaDMAStrided<false, 16, 128, 32, 32> dma(...);
dma.start_async_dma();  // ← Won't work correctly!
```

### 2. Alignment Violations
```cpp
// ❌ WRONG: Data not aligned to 16 bytes
cudaDMAStrided<true, 16, ...> dma(...);
float *misaligned = ptr + 1;  // ← Not 16-byte aligned
dma.execute_dma(misaligned, dst);
```

### 3. Thread Count Mismatch
```cpp
// ❌ WRONG: Block size doesn't match total threads
dim3 block(256, 1);  // 256 threads
cudaDMAStrided<..., 32, 32> dma(...);  // Expects 256 + 64 = 320
```

---

## File Statistics

- **Total Lines**: ~4,217
- **Total Classes**: 35
  - cudaDMASequential: 7
  - cudaDMAStrided: 9
  - cudaDMAIndirect: 18
  - Base classes: 1
- **Warp-Specialized**: ~18
- **Non-Warp-Specialized**: ~17

---

## Comparison with cudaDMAv2.h

| Feature | cudaDMA.h | cudaDMAv2.h |
|---------|-----------|-------------|
| Sequential transfers | ✅ Yes (7 classes) | ❌ No (removed) |
| Strided transfers | ✅ Yes (9 classes) | ✅ Yes (26 classes) |
| Indirect transfers | ✅ Yes (18 classes) | ✅ Yes (similar) |
| Code complexity | 🟢 Simpler | 🟡 More complex |
| Optimization | 🟢 Good | 🟢 Better |
| Use case | General purpose | Performance-critical |

---

## Recommendation

For my GEMM kernel:
- ✅ Using `cudaDMAStrided` is **correct**
- ✅ Using `DO_SYNC=true` is **optimal**
- ✅ Configuration matches my requirements

The library provides the flexibility to optimize for different scenarios while maintaining the warp specialization benefits I need for compute/DMA overlap!
