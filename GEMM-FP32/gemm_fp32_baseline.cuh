/**
 * gemm_fp32.cuh: Modified version of GEMM specifically for FP32 cores
 */

#ifndef GEMM_FP32_CUH
#define GEMM_FP32_CUH

/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET) && !defined(HUGE_DATASET)
#  define STANDARD_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(N)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#define NI 512
#define NJ 512
#define NK 512
#  endif

#  ifdef SMALL_DATASET
#define NI 512
#define NJ 512
#define NK 512
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#define NI 512
#define NJ 512
#define NK 512
#  endif

#  ifdef LARGE_DATASET
#define NI 1024
#define NJ 1024
#define NK 1024
#  endif

#  ifdef EXTRALARGE_DATASET
#define NI 2048
#define NJ 2048
#define NK 2048
#  endif

#  ifdef HUGE_DATASET
#define NI 4096
#define NJ 4096
#define NK 4096
#  endif
# endif /* !N */

# define _PB_NI POLYBENCH_LOOP_BOUND(NI,ni)
# define _PB_NJ POLYBENCH_LOOP_BOUND(NJ,nj)
# define _PB_NK POLYBENCH_LOOP_BOUND(NK,nk)

// Explicitly using float for FP32 operations
typedef float fp32_t;
#define DATA_TYPE fp32_t
#define DATA_PRINTF_MODIFIER "%0.2f "

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Shared memory tile dimensions */
#define TILE_SIZE 32

#endif /* !GEMM_FP32_CUH */