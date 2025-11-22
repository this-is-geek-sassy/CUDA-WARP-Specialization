/**
 * jacobi2D_cudaDMA.cuh: Jacobi 2D stencil with cudaDMA warp specialization
 *
 * Extends baseline with cudaDMAv2 for warp-specialized memory transfers
 */

#ifndef JACOBI2D_CUDADMA_CUH
#define JACOBI2D_CUDADMA_CUH

/* Default to STANDARD_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(STANDARD_DATASET) &&   \
    !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET) && !defined(HUGE_DATASET) && \
    !defined(HUMONGOUS_DATASET)
#define STANDARD_DATASET
#endif

/* Do not define anything if the user manually defines the size. */
#if !defined(N)
/* Define the possible dataset sizes. */
#ifdef MINI_DATASET
#define TSTEPS 20
#define N 128
#endif

#ifdef SMALL_DATASET
#define TSTEPS 40
#define N 256
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
#define TSTEPS 100
#define N 1024
#endif

#ifdef LARGE_DATASET
#define TSTEPS 200
#define N 2048
#endif

#ifdef EXTRALARGE_DATASET
#define TSTEPS 500
#define N 4096
#endif

#ifdef HUGE_DATASET
#define TSTEPS 1000
#define N 8192
#endif

#ifdef HUMONGOUS_DATASET
#define TSTEPS 2000
#define N 16384
#endif
#endif /* !N */

#define _PB_TSTEPS POLYBENCH_LOOP_BOUND(TSTEPS, tsteps)
#define _PB_N POLYBENCH_LOOP_BOUND(N, n)

#ifndef DATA_TYPE
#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Thread block dimensions for baseline kernels */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Shared memory tile dimensions (includes halo) */
#define TILE_X (DIM_THREAD_BLOCK_X + 2) // +2 for left and right halo
#define TILE_Y (DIM_THREAD_BLOCK_Y + 2) // +2 for top and bottom halo

/* cudaDMA configuration */
#define COMPUTE_THREADS_PER_CTA 256 // Compute threads
#define DMA_THREADS_PER_LD 32       // DMA threads per loader (1 warp)
#define NUM_DMA_LOADERS 1           // 1 DMA loader for input array
#define TOTAL_THREADS (COMPUTE_THREADS_PER_CTA + NUM_DMA_LOADERS * DMA_THREADS_PER_LD)

// Compute thread configuration (16x16 compute threads)
#define COMPUTE_THREADS_X 16
#define COMPUTE_THREADS_Y 16

// Tile size for cudaDMA version (includes halo)
// Match GEMM's 32x32 tile pattern for proper cudaDMA alignment
// GEMM uses 32x32 tiles with 128 bytes/row (32 floats × 4 bytes)
#define CUDADMA_TILE_X 32 // 30 compute + 2 halo = 32 total (128 bytes)
#define CUDADMA_TILE_Y 32 // 30 compute + 2 halo = 32 total

// Adjust compute threads for cudaDMA to match tile requirements
#define CUDADMA_COMPUTE_X 30
#define CUDADMA_COMPUTE_Y 30

#endif /* !JACOBI2D_CUDADMA_CUH */
