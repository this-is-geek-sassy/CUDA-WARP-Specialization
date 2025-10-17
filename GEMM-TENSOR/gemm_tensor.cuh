/**
 * gemm_tensor.cuh: Tensor Core implementation using WMMA API
 * Based on the hierarchical tiling approach for Tensor Cores
 */

#ifndef GEMM_TENSOR_CUH
# define GEMM_TENSOR_CUH

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

/* Use half precision for tensor cores */
# ifndef DATA_TYPE
#  define DATA_TYPE half
#  define DATA_PRINTF_MODIFIER "%0.2f "
# endif

/* WMMA tile dimensions - using 16x16x16 tiles for tensor cores */
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

/* Thread block dimensions - each block processes a 64x64 tile */
#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 16

/* Number of warps per block */
#define WARPS_PER_BLOCK 16
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

/* Warp tile dimensions - each warp processes 16x16 */
#define WARP_SIZE_M 16
#define WARP_SIZE_N 16

#endif /* !GEMM_TENSOR */
