/**
 * jacobi2D_baseline.cuh: Baseline Jacobi 2D stencil with shared memory optimization
 *
 * Based on PolyBench/GPU 1.0 test suite
 * Optimized with shared memory for improved performance
 */

#ifndef JACOBI2D_BASELINE_CUH
#define JACOBI2D_BASELINE_CUH

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
#define DATA_PRINTF_MODIFIER "%0.6lf "
#endif

/* Thread block dimensions for baseline kernel */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Shared memory tile dimensions (includes halo) */
#define TILE_X (DIM_THREAD_BLOCK_X + 2) // +2 for left and right halo
#define TILE_Y (DIM_THREAD_BLOCK_Y + 2) // +2 for top and bottom halo

#endif /* !JACOBI2D_BASELINE_CUH */
