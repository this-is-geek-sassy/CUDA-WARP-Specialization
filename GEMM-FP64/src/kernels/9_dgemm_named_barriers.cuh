#ifndef DGEMM_NAMED_BARRIERS_CUH
#define DGEMM_NAMED_BARRIERS_CUH

#include <cuda.h>
#include <cuda/barrier>
#include <cassert>
#include "utils/global_mem_utils.cuh"

/// @brief Bank Conflicts Free DGEMM Kernel
/// @param BM Tile Size Dimension (compile-time constant)
/// @param BK Tile Size Dimension (compile-time constant)
/// @param BN Tile Size Dimension (compile-time constant)
/// @param TM Work per thread across m-dimension (compile-time constant)
/// @param TN Work per thread across n-dimension (compile-time constant)
/// @param TK Work per thread across k-dimension (compile-time constant)
/// @param alpha DGEMM parameter
/// @param beta DGEMM parameter
/// @param M Number of rows in A
/// @param N Number of cols in B
/// @param K Number of cols in A and number of rows in B
/// @param A Pointer to A matrix (M x K)
/// @param B Pointer to B matrix (K x N)
/// @param C Pointer to C matrix (M x N)
template<unsigned int BM, unsigned int BK, unsigned int BN,
         unsigned int TM, unsigned int TN, unsigned int TK,
         unsigned int NUM_TILE_LOAD_WARPS, unsigned int NUM_GLOBAL_LOAD_WARPS, unsigned int WARP_SIZE>
__global__ void dgemm_named_barriers(float alpha, float beta, int M, int N, int K, float* A, float* B, float* C) {
  constexpr unsigned int BDM = (BM/TM);
  constexpr unsigned int BDN = (BN/TN);

  constexpr unsigned int NUM_GLOBAL_LOAD_THREADS = NUM_GLOBAL_LOAD_WARPS * WARP_SIZE; 
  constexpr unsigned int NUM_TILE_LOAD_THREADS = NUM_TILE_LOAD_WARPS * WARP_SIZE; 
  constexpr unsigned int NUM_COMPUTE_THREADS = BDM * BDN; 

  extern __shared__ float sm[];
  float* sA[2] = {&sm[0], &sm[BM * BK]};
  float* sB[2] = {&sm[2 * BM * BK], &sm[2 * BM * BK + BK * BN]};
  float* sC = &sm[2 * (BM * BK + BK * BN)];

  __shared__ cuda::barrier<cuda::thread_scope_block> tile_barrier;
  __shared__ cuda::barrier<cuda::thread_scope_block> global_barrier;

  const unsigned int bm = blockIdx.y * BM;
  const unsigned int bn = blockIdx.x * BN;
  float* gC = C + bm * N + bn;

  unsigned int tId = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned int gId = tId < NUM_GLOBAL_LOAD_THREADS ? 1 : (tId < NUM_GLOBAL_LOAD_THREADS + NUM_TILE_LOAD_THREADS ? 2 : 3);

  if(tId == 0) {
    init(&tile_barrier, NUM_TILE_LOAD_THREADS + NUM_COMPUTE_THREADS);
    init(&global_barrier, NUM_GLOBAL_LOAD_THREADS + NUM_COMPUTE_THREADS);
  }
  __syncthreads();

  if(gId == 1) {
    //// GLOBAL LOAD WARPS
    // Load C from global memory to shared memory.
    readTileChunked<BM, BN, NUM_GLOBAL_LOAD_THREADS, 0>(N, gC, sC);

    // Sync with the compute warps.
    global_barrier.arrive_and_wait();

    // Sync with the entire block.
    __syncthreads();

    // The entire block collectively writes back to gC.
    writeTileChunked<BM, BN, NUM_COMPUTE_THREADS + NUM_TILE_LOAD_THREADS + NUM_GLOBAL_LOAD_THREADS, 0>(N, sC, gC);

  } else if (gId == 2) {
    //// TILE LOAD WARPS
    int buf = 0;
    float* gA = A + bm * K;
    float* gB = B + bn;

    for(unsigned int bk = 0; bk < K; bk += BK) {
      // Load the next tile into extra buffers.
      readTileChunked<BM, BK, NUM_TILE_LOAD_THREADS, NUM_GLOBAL_LOAD_THREADS>(K, gA, sA[buf]);
      readTileChunked<BK, BN, NUM_TILE_LOAD_THREADS, NUM_GLOBAL_LOAD_THREADS>(N, gB, sB[buf]);

      // Update pointers to next tile.
      gA += BK;
      gB += BK * N;

      // Swap buffers.
      buf ^= 1;

      // Sync with the compute warps.
      tile_barrier.arrive_and_wait();
    }

    // Sync with the entire block.
    __syncthreads();

    // The entire block collectively writes back to gC.
    writeTileChunked<BM, BN, NUM_COMPUTE_THREADS + NUM_TILE_LOAD_THREADS + NUM_GLOBAL_LOAD_THREADS, 0>(N, sC, gC);

  } else {
    //// COMPUTE WARPS
    tId -= NUM_GLOBAL_LOAD_THREADS + NUM_TILE_LOAD_THREADS;
    const unsigned int tx = tId % BDN;
    const unsigned int ty = tId / BDN;

    int mem = 0;
    float acc_reg[TM][TN];

    for(int i = 0; i < TM; i++)
      for(int j = 0; j < TN; j++)
        acc_reg[i][j] = 0.0;

    for(unsigned int bk = 0; bk < K; bk += BK) {
      // Sync with the tile load warps.
      tile_barrier.arrive_and_wait();

      // Process the current tile.
      for(int k = 0; k < BK; k++)
        for(int i = 0; i < TM; i++)
          for(int j = 0; j < TN; j++)
            acc_reg[i][j] = fma(sA[mem][(ty + i * BDM) * BK + k], sB[mem][k * BN + (tx + j * BDN)], acc_reg[i][j]);

      // Swap buffers.
      mem ^= 1;
    }

    // Sync with the global load warps.
    global_barrier.arrive_and_wait();

    // Epilogue
    for(int i = 0; i < TM; i++) {
      for(int j = 0; j < TN; j++) {
        sC[(ty + i * BDM) * BN + tx + j * BDN] = alpha * acc_reg[i][j] + beta * sC[(ty + i * BDM) * BN + tx + j * BDN];
      }
    }

    // Sync with the entire block.
    __syncthreads();

    // The entire block collectively writes back to gC.
    writeTileChunked<BM, BN, NUM_COMPUTE_THREADS + NUM_TILE_LOAD_THREADS + NUM_GLOBAL_LOAD_THREADS, 0>(N, sC, gC);
  }
}

#endif // DGEMM_NAMED_BARRIERS_CUH