/**
 * gemm_tensor.cuh: Tensor Core implementation using WMMA API
 * Based on the hierarchical tiling approach for Tensor Cores
 */

#ifndef GEMM_TENSOR_CUH
# define GEMM_TENSOR_CUH

/* Default to LARGE_DATASET (1024×1024). */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET) && !defined(HUGE_DATASET)
#  define EXTRALARGE_DATASET
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

#  ifdef STANDARD_DATASET 
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
# endif 

# define _PB_NI POLYBENCH_LOOP_BOUND(NI,ni)
# define _PB_NJ POLYBENCH_LOOP_BOUND(NJ,nj)
# define _PB_NK POLYBENCH_LOOP_BOUND(NK,nk)


# ifndef DATA_TYPE
#  define DATA_TYPE half
#  define DATA_PRINTF_MODIFIER "%0.2f "
# endif

/* WMMA tile dimensions - using 16x16x16 tiles for tensor cores */
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

/* Thread block dimensions - rectangular tile for better occupancy
 * Using 128x64 tile: more output elements per block, better parallelism */
#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 16

/* Number of warps per block (512 threads = 16 warps) */
#define WARPS_PER_BLOCK 16
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

/* Warp tile dimensions - each warp still processes 16x16 WMMA tiles
 * With 128x64 block and 16x16 warp tiles: 8x4 = 32 warp tiles per block
 * But we only have 16 warps, so each warp handles 2 tiles */
#define WARP_SIZE_M 16
#define WARP_SIZE_N 16

/* Number of WMMA tiles per block dimension */
#define BLOCK_TILES_M (BLOCK_SIZE_M / WMMA_M)  // 128/16 = 8 tiles
#define BLOCK_TILES_N (BLOCK_SIZE_N / WMMA_N)  // 64/16 = 4 tiles
#define TILES_PER_WARP 2  // Each warp handles 2 output tiles

/* For warp specialization: use 4 DMA warps, 12 compute warps */
#define NUM_LOAD_WARPS 4
#define NUM_COMPUTE_WARPS (WARPS_PER_BLOCK - NUM_LOAD_WARPS)

/* Thread work division for loading (used by load warps) */
#define ELEMS_PER_LOAD_THREAD 2  // Load 2 half values (4 bytes) per iteration

#endif /* !GEMM_TENSOR */
