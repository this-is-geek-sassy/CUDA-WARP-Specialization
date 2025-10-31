#ifndef DGEMM_8_WARP_SPECIALIZED_CUH
#define DGEMM_8_WARP_SPECIALIZED_CUH

#include <cuda.h>
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
         unsigned int NUM_LOAD_WARPS, unsigned int WARP_SIZE>
__global__ void dgemm_warp_specialized(float alpha, float beta, int M, int N, int K, float* A, float* B, float* C) {
  constexpr unsigned int BDM = (BM/TM); // blockDim.y (compile time constant)
  constexpr unsigned int BDN = (BN/TN); // blockDim.x (compile time constant)

  extern __shared__ float sm[];
  float* sA[2] = {&sm[0], &sm[BM * BK]};
  float* sB[2] = {&sm[2 * BM * BK], &sm[2 * BM * BK + BK * BN]};

  constexpr unsigned int NUM_LOAD_THREADS = NUM_LOAD_WARPS * WARP_SIZE; 

  const unsigned int bm = blockIdx.y * BM;
  const unsigned int bn = blockIdx.x * BN;

  unsigned int tId = threadIdx.y * blockDim.x + threadIdx.x;
  const bool isLoadWarp = (tId / WARP_SIZE) < NUM_LOAD_WARPS;

  if(isLoadWarp) {
    //// LOAD WARPS
    int buf = 0;
    float* gA = A + bm * K;
    float* gB = B + bn;

    for(unsigned int bk = 0; bk < K; bk += BK) {
      // Load the next tile into extra buffers.
      readTileChunked<BM, BK, NUM_LOAD_THREADS, 0>(K, gA, sA[buf]);
      readTileChunked<BK, BN, NUM_LOAD_THREADS, 0>(N, gB, sB[buf]);

      // Update pointers to next tile.
      gA += BK;
      gB += BK * N;

      // Swap buffers.
      buf ^= 1;

      __syncthreads(); // Sync with the compute warps.
    }
  }
  else {
    //// COMPUTE WARPS
    tId -= NUM_LOAD_THREADS;
    const unsigned int tx = tId % BDN;
    const unsigned int ty = tId / BDN;

    int mem = 0;
    float acc_reg[TM][TN];

    for(int i = 0; i < TM; i++)
      for(int j = 0; j < TN; j++)
        acc_reg[i][j] = 0.0;

    for(unsigned int bk = 0; bk < K; bk += BK) {
      __syncthreads(); // Sync with the load warps.

      for(int k = 0; k < BK; k++)
        for(int i = 0; i < TM; i++)
          for(int j = 0; j < TN; j++)
            acc_reg[i][j] = fma(sA[mem][(ty + i * BDM) * BK + k], sB[mem][k * BN + (tx + j * BDN)], acc_reg[i][j]);

      mem ^= 1; // Swap Buffers
    }

    // Epilogue
    for(int i = 0; i < TM; i++) {
      for(int j = 0; j < TN; j++) {
        C[(bm + ty + i * BDM) * N + (bn + tx + j * BDN)] = alpha * acc_reg[i][j] + beta * C[(bm + ty + i * BDM) * N + (bn + tx + j * BDN)];
      }
    }
  }
}

#endif // DGEMM_8_WARP_SPECIALIZED_CUH