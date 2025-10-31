# cudaDMAStrided quick reference

This note explains how to instantiate and use `cudaDMAStrided` for warp-specialized DMA loading in CUDA kernels. It summarizes the template parameters and the constructor arguments you pass at runtime.

## Template parameters

`cudaDMAStrided<DO_SYNC, ALIGNMENT, BYTES_PER_ELMT, DMA_THREADS, NUM_ELMTS>`

- DO_SYNC (bool)
  - true: compute and DMA threads synchronize via barriers inside `execute_dma`. Use when DMA and compute are in the same CTA and cooperate via `start_async_dma()`, `wait_for_dma_finish()`.
  - false: no implicit sync in the transfer primitive. Use only if you manage synchronization externally (advanced).

- ALIGNMENT (int, one of 4, 8, 16)
  - Vectorization width in bytes used for bulk moves (float, float2, float4). Pick based on your element size and alignment guarantees.

- BYTES_PER_ELMT (int)
  - Size in bytes of one logical element that each DMA thread-group transfers per row/stripe before advancing by the row stride. In this API, an "element" is a contiguous chunk of bytes within a single row of the source that maps to a single row of the destination tile.
  - In tiled GEMM, the conventional choice is: a single tile row worth of data.
    - Example (float GEMM with TILE_SIZE=32): one tile row holds 32 floats; `BYTES_PER_ELMT = 32 * sizeof(float) = 128`.
    - For double (TILE_SIZE=32): `BYTES_PER_ELMT = 32 * sizeof(double) = 256`.
  - Other patterns are possible as long as your notion of “element” is consistent with `NUM_ELMTS` and the provided strides:
    - If you copy a partial row (e.g., 20 floats): `BYTES_PER_ELMT = 20 * sizeof(float)` and `NUM_ELMTS` is the number of such partial rows to move.
    - If you copy a column into a column-major destination tile, then your element could be a column segment; `src_stride`/`dst_stride` must be set accordingly.

- DMA_THREADS (int)
  - Number of threads in the block assigned to DMA (i.e., loading warps). Commonly 32 or multiples of 32 per loader.

- NUM_ELMTS (int)
  - Number of elements (e.g., rows) to transfer per execute step. For a 32x32 tile, this is 32.

Specializations provided in `cudaDMA.h` allow partially specifying these at runtime:

- `cudaDMAStrided<DO_SYNC, ALIGNMENT, BYTES_PER_ELMT, DMA_THREADS, NUM_ELMTS>`: all compile-time; fastest.
- `cudaDMAStrided<DO_SYNC, ALIGNMENT, 0, 0, 0>`: BYTES_PER_ELMT/NUM_ELMTS/DMA_THREADS taken at runtime by constructor.
- `cudaDMAStrided<DO_SYNC, ALIGNMENT>` (alias of above form in the header): accepts sizes at runtime; slightly more flexible.

## Constructors (runtime arguments)

There are two constructor shapes depending on whether source and destination pitches are the same.

1) Single-stride (destination stride == BYTES_PER_ELMT)

- When BYTES_PER_ELMT equals the destination row stride, use the shorter ctor:

For the fully compile-time form:

```
__device__ cudaDMAStrided(
    int dmaID,
    int num_compute_threads,
    int dma_threadIdx_start,
    int el_stride);
```

For the runtime-sized variant (`BYTES_PER_ELMT/DMA_THREADS/NUM_ELMTS` provided at runtime):

```
__device__ cudaDMAStrided(
    int dmaID,             // Unique barrier domain (two barriers are consumed per ID)
    int DMA_THREADS,       // Number of DMA threads in this CTA
    int num_compute_threads,
    int dma_threadIdx_start, // First threadIdx.x owned by this DMA loader
    int BYTES_PER_ELMT,    // Size of one element in bytes
    int NUM_ELMTS,         // Elements per transfer step (e.g., tile height)
    int el_stride          // Source and destination row stride in bytes
);
```

2) Dual-stride (explicit source and destination strides)

Use when source pitch differs from destination pitch (common when copying from global pitched layout into a packed shared tile):

Compile-time sized form:

```
__device__ cudaDMAStrided(
    int dmaID,
    int num_compute_threads,
    int dma_threadIdx_start,
    int src_stride,        // Source row stride in bytes
    int dst_stride         // Destination row stride in bytes
);
```

Runtime-sized variant:

```
__device__ cudaDMAStrided(
    int dmaID,
    int DMA_THREADS,
    int num_compute_threads,
    int dma_threadIdx_start,
    int BYTES_PER_ELMT,
    int NUM_ELMTS,
    int src_stride,
    int dst_stride);
```

Notes on key parameters:
- dmaID: selects the pair of named barriers used to coordinate with compute threads. Each loader must have a unique `dmaID` within the CTA.
- num_compute_threads: threads in the block that are not DMA threads; they will call `start_async_dma()` and `wait_for_dma_finish()`.
- dma_threadIdx_start: first threadIdx.x that belongs to this DMA loader; range is `[start, start+DMA_THREADS)`.
- BYTES_PER_ELMT: logical row size to copy per "element". Often equals `TILE_SIZE * sizeof(T)`.
- NUM_ELMTS: number of rows (or columns if transposed) to copy per tile.
- src_stride/dst_stride/el_stride: pitch in bytes between successive elements in source/destination.

## Methods used in kernels

- `bool owns_this_thread() const` — true for threads that belong to the DMA loader.
- `void start_async_dma() const` — called by compute threads to signal the DMA threads to start the next load.
- `void wait_for_dma_finish() const` — called by compute threads to wait until the current load completes.
- `void execute_dma(const void* src, void* dst) const` — called by DMA threads to perform the copy from `src` to `dst` using the configured striding and alignment.

## Minimal usage pattern (two loaders: A and B)

```cuda
// Shared tiles
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// DMA loaders: one warp each
const int DMA_THREADS_PER_LD = 32;             // one warp
const int COMPUTE_THREADS = 256;               // e.g., 8 warps for compute
const int BYTES_PER_ROW = TILE_SIZE * sizeof(float); // 32 * 4 = 128

cudaDMAStrided<true, 16, BYTES_PER_ROW, DMA_THREADS_PER_LD, TILE_SIZE>
  dmaA(/*dmaID=*/0,
       /*num_compute_threads=*/COMPUTE_THREADS,
       /*dma_threadIdx_start=*/COMPUTE_THREADS,
       /*src_stride=*/nk * sizeof(float),
       /*dst_stride=*/BYTES_PER_ROW);

cudaDMAStrided<true, 16, BYTES_PER_ROW, DMA_THREADS_PER_LD, TILE_SIZE>
  dmaB(/*dmaID=*/1,
       /*num_compute_threads=*/COMPUTE_THREADS,
       /*dma_threadIdx_start=*/COMPUTE_THREADS + DMA_THREADS_PER_LD,
       /*src_stride=*/nj * sizeof(float),
       /*dst_stride=*/BYTES_PER_ROW);

if (threadIdx.x < COMPUTE_THREADS) {
  // compute threads
  dmaA.start_async_dma();
  dmaB.start_async_dma();
  for (int t = 0; t < numTiles; ++t) {
    dmaA.wait_for_dma_finish();
    dmaB.wait_for_dma_finish();
    // ... compute using As, Bs ...
    if (t+1 < numTiles) { dmaA.start_async_dma(); dmaB.start_async_dma(); }
  }
} else if (dmaA.owns_this_thread()) {
  for (int t = 0; t < numTiles; ++t) {
    const float* src = &A[(by*TILE_SIZE) * nk + t*TILE_SIZE];
    dmaA.execute_dma(src, As);
  }
} else if (dmaB.owns_this_thread()) {
  for (int t = 0; t < numTiles; ++t) {
    const float* src = &B[(t*TILE_SIZE) * nj + bx*TILE_SIZE];
    dmaB.execute_dma(src, Bs);
  }
}
```

## What is an “element” here?

In `cudaDMAStrided`, the term element does not mean a scalar like one float. It refers to the atomic chunk that the DMA layer treats as one row unit before adding the row stride. Practically:

- element = a contiguous span of bytes copied as one unit for a given row step.
- `BYTES_PER_ELMT` = byte size of that span.
- `NUM_ELMTS` = how many such spans (rows/segments) you copy in one tile.

Mapping to tiled GEMM:
- When you load a TILE_SIZE × TILE_SIZE block from global to shared, it’s common to define:
  - element = one tile row
  - `BYTES_PER_ELMT = TILE_SIZE * sizeof(T)`
  - `NUM_ELMTS = TILE_SIZE`
- The source stride is the pitch in bytes between the starts of consecutive rows in global memory (e.g., `lda * sizeof(T)`); destination stride is usually `BYTES_PER_ELMT` for a tightly packed shared tile.

Quick checklist:
- Can I describe my transfer as NUM_ELMTS repetitions of copying BYTES_PER_ELMT bytes separated by src_stride bytes in source and dst_stride bytes in destination? If yes, your element definition is consistent.
- If you change tile width or data type, update BYTES_PER_ELMT accordingly.
- If you transpose or pack/unpack, make sure strides reflect the layout and element still refers to the contiguous chunk per step.

## Best practices and gotchas

- ALIGNMENT must be 4, 8, or 16. Choose the largest that respects your data alignment; misalignment degrades or breaks vector loads.
- Ensure `DMA_THREADS` is a multiple of 32; each loader should map to whole warps.
- `dmaID` must be unique per loader within the block; each consumes two named barriers.
- The compute region must call `start_async_dma()` before `wait_for_dma_finish()` in each iteration.
- Keep source/destination pointers appropriately cast and pitched; strides are in BYTES.
- When tiling, BYTES_PER_ELMT typically equals the width-in-elements of the tile multiplied by sizeof(element).
- If you change tile sizes at runtime, prefer the runtime-sized `cudaDMAStrided<true, ALIGNMENT>` form and pass sizes via the constructor.

## Mapping to the header definitions

- The README examples match the macros used internally (`SINGLE_STRIDED_BASE`, `DOUBLE_STRIDED_BASE`) and the available specializations defined around the `cudaDMAStrided` declarations in `cudaDMA.h`.
- See also the related `cudaDMAIndirect` class for gather/scatter patterns.
