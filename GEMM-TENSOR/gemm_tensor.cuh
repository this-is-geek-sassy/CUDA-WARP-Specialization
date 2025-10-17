/**
 * gemm_tensor.cuh: Tensor Core implementation using WMMA API
 * Based on the hierarchical tiling approach for Tensor Cores
 */

#ifndef GEMM_TENSOR_CUH
# define GEMM_TENSOR_CUH


# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define STANDARD_DATASET
# endif


# if !defined(N)

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
# endif /* !N */

# define _PB_NI POLYBENCH_LOOP_BOUND(NI,ni)
# define _PB_NJ POLYBENCH_LOOP_BOUND(NJ,nj)
# define _PB_NK POLYBENCH_LOOP_BOUND(NK,nk)


# ifndef DATA_TYPE
#  define DATA_TYPE half
#  define DATA_PRINTF_MODIFIER "%0.2f "
# endif


#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 16


#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)


#define WARP_SIZE_M 64
#define WARP_SIZE_N 64

#endif
