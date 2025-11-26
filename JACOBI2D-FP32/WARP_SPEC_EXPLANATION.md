# Understanding Coalesced Memory Access in Warp Specialization

## What I Implemented

I used **coalesced memory access** in the warp specialization kernel to maximize memory bandwidth. This document explains how and why I did it.

---

## The Problem with Normal Memory Access

When threads access memory randomly, each thread causes a separate memory transaction. This is slow.

```cuda
// BAD - Each thread loads from different location
int random_idx = (threadIdx.x * 17) % 612;
data = A[random_idx];  // 32 threads = 32 separate memory transactions!
```

This kills performance because the GPU has to make 32 individual requests to memory.

---

## My Solution: Coalesced Strided Loading

I made threads access consecutive memory addresses so the GPU can combine them into one transaction.

Here's the key part of my code:

```cuda
// From jacobi2D_warp_spec.cu, lines 108-120
for (int idx = tid; idx < TILE_SIZE; idx += NUM_PRODUCERS) {
    int tile_y = idx / TILE_WIDTH;
    int tile_x = idx % TILE_WIDTH;
    
    int global_y = blockIdx.y * DIM_THREAD_BLOCK_Y + tile_y - 1;
    int global_x = blockIdx.x * DIM_THREAD_BLOCK_X + tile_x - 1;
    
    if (global_y >= 0 && global_y < n && global_x >= 0 && global_x < n) {
        tile[tile_y][tile_x] = A[global_y * N + global_x];
    }
}
```

Let me break down why this works.

## Line-by-Line Breakdown

### Step 1: Loop Setup
```cuda
for (int idx = tid; idx < TILE_SIZE; idx += NUM_PRODUCERS)
```

- `tid` is my thread ID (0 to 511)
- `TILE_SIZE` is 612 (the total elements I need to load: 34×18)
- `NUM_PRODUCERS` is 128 (only first 4 warps are loading)

**What this does:** Each thread starts at a different index and jumps by 128 each iteration.

- Thread 0 loads: idx = 0, 128, 256, 384, 512
- Thread 1 loads: idx = 1, 129, 257, 385, 513  
- Thread 2 loads: idx = 2, 130, 258, 386, 514
- ...
- Thread 31 loads: idx = 31, 159, 287, 415, 543

Notice how threads 0-31 load consecutive indices (0,1,2...31). This is coalescing!

---

### Step 2: Convert to 2D Position
```cuda
int tile_y = idx / TILE_WIDTH;
int tile_x = idx % TILE_WIDTH;
```

- `TILE_WIDTH` is 34
- This converts linear index to 2D coordinates in my shared memory tile

Example: If `idx = 35`, then `tile_y = 1` and `tile_x = 1` (second row, second column)

---

### Step 3: Map to Global Memory
```cuda
int global_y = blockIdx.y * DIM_THREAD_BLOCK_Y + tile_y - 1;
int global_x = blockIdx.x * DIM_THREAD_BLOCK_X + tile_x - 1;
```

- I need to account for which block I'm in (`blockIdx`)
- The `-1` is because I'm loading halo cells (one extra row/column around edges)
- This gives me the actual row and column in the 4096×4096 global array

---

### Step 4: Load from Global Memory
```cuda
if (global_y >= 0 && global_y < n && global_x >= 0 && global_x < n) {
    tile[tile_y][tile_x] = A[global_y * N + global_x];
}
```

- Bounds check to avoid loading outside array
- `A[global_y * N + global_x]` is the global memory address
- Store into my shared memory tile at `[tile_y][tile_x]`

**The key:** When threads 0-31 do this in the first iteration (idx = 0-31), they access:
- `A[base + 0]`, `A[base + 1]`, `A[base + 2]`, ... `A[base + 31]`

These are consecutive addresses, so GPU combines into ONE memory transaction!

---

The **Jacobi update rule** is:

$$
B[i][j] = \frac{1}{4} \left( A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1] \right)
$$

## Why Coalescing Matters

Without coalescing, my kernel would be 20× slower at loading data!

**With coalescing (my implementation):**
- 612 elements ÷ 128 threads = ~5 elements per thread
- 5 iterations × 1 memory transaction per iteration = 5 transactions total per warp
- Load time: ~5 × 200 cycles = 1000 cycles

**Without coalescing (random access):**
- Same 612 elements
- But each of 128 threads causes separate transaction
- 5 iterations × 32 transactions per iteration = 160 transactions per warp
- Load time: ~160 × 200 cycles = 32,000 cycles

That's why I carefully designed the strided access pattern.

---

## Verification: How to Check Coalescing

You can verify this with `nvprof` or `nsight compute`:

```bash
nvprof --metrics gld_efficiency ./jacobi2d_warp_spec
```

Look for "Global Load Efficiency" close to 100% - that means coalescing is working.

In my case, the strided pattern ensures consecutive threads access consecutive addresses, giving near-perfect coalescing.

---

## Summary

I used coalesced memory access because:

1. **Simple pattern:** `idx = tid; idx += 128` ensures threads 0-31 hit consecutive addresses
2. **Hardware optimization:** GPU combines 32 individual loads into 1 transaction
3. **20× speedup:** Reduces memory transactions from 160 to 5 per warp
4. **Proven technique:** Standard approach in CUDA optimization

The key insight: consecutive threads should access consecutive memory locations. My strided loading achieves this automatically.

---

## Code Location

Check `jacobi2D_warp_spec.cu` lines 108-120 to see the full implementation.
