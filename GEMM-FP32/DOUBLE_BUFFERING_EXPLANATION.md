# Double Buffering Implementation in cudaDMA GEMM

## Overview

The cudaDMA GEMM kernel now uses **double buffering** to overlap computation with data transfer, maximizing GPU utilization.

## Architecture

### Shared Memory Layout

```
Before (Single Buffer):
┌─────────────────────┐
│  As [32x32]         │  ← Only one buffer for A
├─────────────────────┤
│  Bs [32x32]         │  ← Only one buffer for B
└─────────────────────┘

After (Double Buffer):
┌─────────────────────┐
│  As_0 [32x32]       │  ← Buffer 0 for A
├─────────────────────┤
│  Bs_0 [32x32]       │  ← Buffer 0 for B
├─────────────────────┤
│  As_1 [32x32]       │  ← Buffer 1 for A
├─────────────────────┤
│  Bs_1 [32x32]       │  ← Buffer 1 for B
└─────────────────────┘
```

**Memory Increase**: 2x (from 8KB to 16KB of shared memory per block)

### Thread Organization

- **256 Compute Threads**: Perform matrix multiplication
- **32 DMA Threads (Warp A)**: Load matrix A tiles
- **32 DMA Threads (Warp B)**: Load matrix B tiles
- **Total**: 320 threads per block

## Double Buffering Pipeline

### Timeline Visualization

```
Time →
Tile:        0          1          2          3
         ┌─────────┬─────────┬─────────┬─────────┐
DMA A:   │Load→As_0│Load→As_1│Load→As_0│Load→As_1│
         └─────────┴─────────┴─────────┴─────────┘
         ┌─────────┬─────────┬─────────┬─────────┐
DMA B:   │Load→Bs_0│Load→Bs_1│Load→Bs_0│Load→Bs_1│
         └─────────┴─────────┴─────────┴─────────┘
         ┌─────────┬─────────┬─────────┬─────────┐
Compute: │  Wait   │Use As_0 │Use As_1 │Use As_0 │
         │         │Use Bs_0 │Use Bs_1 │Use Bs_0 │
         └─────────┴─────────┴─────────┴─────────┘
```

### Iteration Flow

**Iteration t = 0:**
1. DMA threads load tile 0 → As_0, Bs_0
2. Compute threads wait for load to complete
3. Compute threads start computation on As_0, Bs_0
4. DMA threads simultaneously load tile 1 → As_1, Bs_1

**Iteration t = 1:**
1. Compute threads wait for tile 1 load to complete
2. Compute threads start computation on As_1, Bs_1
3. DMA threads simultaneously load tile 2 → As_0, Bs_0

**Iteration t = 2:**
1. Compute threads wait for tile 2 load to complete
2. Compute threads start computation on As_0, Bs_0
3. DMA threads simultaneously load tile 3 → As_1, Bs_1

And so on...

## Key Implementation Details

### Buffer Selection (Ping-Pong)

```cpp
int curr_buf = t & 1;  // Current buffer index: 0 or 1
```

- **Even iterations** (t = 0, 2, 4, ...): Use buffer 0
- **Odd iterations** (t = 1, 3, 5, ...): Use buffer 1

### Compute Thread Logic

```cpp
for (int t = 0; t < numTiles; t++) {
    int curr_buf = t & 1;
    
    // Wait for current tile
    dma_ld_a.wait_for_dma_finish();
    dma_ld_b.wait_for_dma_finish();
    
    // Signal next tile load (overlapped with computation)
    if (t < numTiles - 1) {
        dma_ld_a.start_async_dma();
        dma_ld_b.start_async_dma();
    }
    
    // Compute on current buffer
    if (curr_buf == 0) {
        // Use As_0, Bs_0
    } else {
        // Use As_1, Bs_1
    }
}
```

### DMA Thread Logic

```cpp
for (int t = 0; t < numTiles; t++) {
    int buf_idx = t & 1;
    
    if (buf_idx == 0) {
        dma_ld_a.execute_dma(src_ptr, As_0);
    } else {
        dma_ld_a.execute_dma(src_ptr, As_1);
    }
}
```

## Benefits

### 1. Overlapped Execution
- **Before**: Sequential (load tile → compute → load tile → compute)
- **After**: Overlapped (load tile N+1 WHILE computing on tile N)

### 2. Reduced Idle Time
- Compute threads don't wait idle during loads
- DMA threads don't wait idle during computation

### 3. Better Memory Bandwidth Utilization
- Global memory transfers happen concurrently with computation
- L2 cache can prefetch data while previous tile is being processed

## Performance Considerations

### Pros
✅ Higher GPU utilization  
✅ Reduced memory stall time  
✅ Better overlapping of compute and memory operations  
✅ Same number of DMA objects (reused for alternating buffers)

### Cons
❌ 2x shared memory usage (8KB → 16KB per block)  
❌ May reduce occupancy on GPUs with limited shared memory  
❌ Slightly more complex control flow

### Occupancy Impact

For **32x32 tiles** with **float32**:
- Single buffer: 4KB (As) + 4KB (Bs) = **8KB** per block
- Double buffer: 8KB (As_0, As_1) + 8KB (Bs_0, Bs_1) = **16KB** per block

On modern GPUs (48KB shared memory per SM):
- Single buffer: 6 blocks per SM (48KB / 8KB)
- Double buffer: 3 blocks per SM (48KB / 16KB)

**Trade-off**: Reduced occupancy but better memory-compute overlap.

## Expected Speedup

Double buffering typically provides **10-30% speedup** for memory-bound kernels, especially when:
- Memory latency is significant
- Computation time ≈ Memory transfer time
- GPU has sufficient shared memory for reduced occupancy

## Verification

The implementation maintains correctness:
- Same compute logic (just alternating buffers)
- Same synchronization points
- Same DMA objects (reused for both buffers)
- Results should match single-buffer version exactly

## Future Optimizations

1. **Triple buffering**: For even better overlap
2. **Adaptive buffering**: Choose single/double based on problem size
3. **Register blocking**: Reduce shared memory footprint per buffer
