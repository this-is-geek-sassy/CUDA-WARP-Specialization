# Performance Measurement Considerations for Jacobi 2D Stencil

## Overview

This document explains critical pitfalls in measuring GPU kernel performance for iterative stencil computations like Jacobi 2D, and how we address them in this implementation.

## The Pitfall: Including Auxiliary Operations in Timing

### Problem Description

In iterative stencil algorithms, each iteration typically requires:

1. **Main computation kernel** (the actual stencil operation)
2. **Copy kernel** (to swap buffers: A ← B)
3. **Memory operations** (for texture memory: updating texture arrays)

**CRITICAL MISTAKE**: Including copy kernels and memory operations in performance measurements leads to **misleading and unfair comparisons** between different optimization strategies.

### Why This Matters

#### Example: Comparing Baseline vs Shared Memory

If we measure the entire iteration loop:

```cuda
// WRONG APPROACH - Measures everything
polybench_start_instruments;
for (int t = 0; t < tsteps; t++) {
    stencil_kernel<<<>>>(...);          // ← What we want to measure
    cudaDeviceSynchronize();
    copy_kernel<<<>>>(...);             // ← NOT what we want to measure
    cudaDeviceSynchronize();
}
polybench_stop_instruments;
```

**Problems:**

- Copy kernel time is **identical** across all implementations
- Including it dilutes the actual optimization gains
- Makes fast kernels appear slower than they are (relative to baseline)
- Hides true performance differences

#### Example: Texture Memory Overhead

For texture memory implementations:

```cuda
// WRONG APPROACH - Includes texture update overhead
polybench_start_instruments;
for (int t = 0; t < tsteps; t++) {
    texture_kernel<<<>>>(...);          // ← What we want to measure
    cudaDeviceSynchronize();
    copy_kernel<<<>>>(...);             // ← NOT what we want to measure
    cudaMemcpy2DToArray(...);          // ← Expensive texture update (NOT what we want to measure)
}
polybench_stop_instruments;
```

**Impact:**

- `cudaMemcpy2DToArray` is **very expensive** and dominates timing
- Makes texture memory appear much slower than it actually is
- Doesn't reflect the actual kernel computation performance
- Penalizes texture approach unfairly

## Our Solution: Measure Only Kernel Execution

### Correct Approach

We measure **only the stencil computation kernel** and accumulate times across iterations:

```cuda
double total_kernel_time = 0.0;
for (int t = 0; t < tsteps; t++) {
    // Measure ONLY the computation kernel
    polybench_start_instruments;
    stencil_kernel<<<grid, block>>>(n, A_gpu, B_gpu);
    cudaDeviceSynchronize();
    polybench_stop_instruments;
    total_kernel_time += polybench_t_end - polybench_t_start;

    // Copy kernel - NOT measured
    copy_kernel<<<grid, block>>>(n, A_gpu, B_gpu);
    cudaDeviceSynchronize();

    // Texture update - NOT measured (for texture versions only)
    // cudaMemcpy2DToArray(...);
}

printf("Total kernel execution time: %0.6lf\n", total_kernel_time);
```

### Benefits

1. **Fair comparison**: All implementations measured on equal footing
2. **Pure optimization impact**: Shows only the actual kernel improvements
3. **No auxiliary overhead**: Copy and memory operations excluded
4. **True performance**: Reflects actual stencil computation speed

## Implementation Details

### What We Measure (Per Iteration)

✅ **Stencil computation kernel execution time**

- Baseline: `jacobi2D_kernel_baseline`
- Shared Memory: `jacobi2D_kernel_shared`
- Texture: `jacobi2D_kernel_texture`
- Hybrid: `jacobi2D_kernel_texture_shared`

### What We DON'T Measure (Per Iteration)

❌ Copy kernel (`jacobi2D_kernel_copy`) execution time
❌ Texture array update (`cudaMemcpy2DToArray`) time
❌ Any memory transfers or synchronization overhead

### Timing Mechanism

```cuda
polybench_start_instruments;  // Start timer
<KERNEL><<<grid, block>>>(...);
cudaDeviceSynchronize();      // Ensure kernel completion
polybench_stop_instruments;   // Stop timer
total_time += (polybench_t_end - polybench_t_start);
```

## Performance Comparison Guidelines

### Interpreting Results

1. **Baseline (Global Memory Only)**

   - Direct reads from global memory
   - No caching optimization
   - Reference point for comparisons

2. **Shared Memory**

   - Explicit tile-based caching
   - Halo region loading
   - **Expected: 2-3x faster than baseline**
   - Best for stencil computations

3. **Texture Memory**

   - Hardware-managed texture cache
   - 2D spatial locality
   - **Expected: Similar to baseline or slightly slower**
   - Texture update overhead NOT in measurement, but affects overall app performance

4. **Hybrid (Texture + Shared)**
   - Texture reads → Shared memory tiles
   - Combined caching benefits
   - **Expected: Similar to pure shared memory**
   - May have slight overhead from texture reads

### Real-World Considerations

While we measure only kernel time, in production:

- Copy kernels add ~constant overhead
- Texture updates add significant overhead for texture-based methods
- Total application time = kernel time + auxiliary operations
- For fair optimization comparison, use kernel-only measurements
- For deployment decisions, consider total application time

## Key Takeaways

1. **Always measure what you're optimizing**: Kernel computation, not auxiliary operations
2. **Separate concerns**: Kernel performance ≠ Total application performance
3. **Fair comparisons**: Use identical measurement methodology across implementations
4. **Document overhead**: Note auxiliary operations even if not measured
5. **Context matters**: Kernel-only time for optimization analysis, total time for deployment

## Validation

All kernel implementations are validated against CPU reference to ensure correctness:

- Baseline vs CPU
- Shared Memory vs CPU
- Texture vs CPU
- Hybrid vs CPU

Error threshold: 0.05% difference acceptable

---

**Note**: This measurement methodology follows GPU performance benchmarking best practices where kernel execution time is the primary metric for optimization evaluation, while auxiliary operations are considered separately for total application analysis.
