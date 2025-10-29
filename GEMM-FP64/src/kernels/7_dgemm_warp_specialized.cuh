#ifndef DGEMM_WARP_SPECIALIZED_CUH
#define DGEMM_WARP_SPECIALIZED_CUH

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
  float* sC[2] = {&sm[2 * (BM * BK + BK * BN)], &sm[2 * (BM * BK + BK * BN) + BDM * BDN]};

  constexpr unsigned int NUM_LOAD_THREADS = NUM_LOAD_WARPS * WARP_SIZE; 

  const unsigned int bm = blockIdx.y * BM;
  const unsigned int bn = blockIdx.x * BN;

  unsigned int tId = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned int wid = tId / WARP_SIZE;
  const bool isLoadWarp = wid < NUM_LOAD_WARPS;

  if(isLoadWarp) {
    //// LOAD WARPS
    int buf = 0;
    float* gA = A + bm * K;
    float* gB = B + bn;
    float* gC = C + bm * N + bn;

    // Load the first tile.
    readTileChunked<BM, BK, NUM_LOAD_THREADS>(K, gA, sA[buf]);
    readTileChunked<BK, BN, NUM_LOAD_THREADS>(N, gB, sB[buf]);

    // Update pointers to next tile.
    gA += BK;
    gB += BK * N;
    buf = 1 - buf; // Swap buffers.

    __syncthreads(); // Sync with the compute warps.

    for(unsigned int bk = BK; bk < K; bk += BK) {
      // Load the next tile into extra buffers.
      readTileChunked<BM, BK, NUM_LOAD_THREADS>(K, gA, sA[buf]);
      readTileChunked<BK, BN, NUM_LOAD_THREADS>(N, gB, sB[buf]);

      // Update pointers to next tile.
      gA += BK;
      gB += BK * N;

      // Swap buffers.
      buf = 1 - buf;

      __syncthreads(); // Sync with the compute warps.
    }

    for(int i = 0; i < TM; i++) {
      for(int j = 0; j < TN; j++) {
        // Load next tile from C.
        readTileBatched<BDM, BDN, NUM_LOAD_THREADS>(N, gC, sC[buf]);

        gC += BDN;
        buf = 1 - buf; // Swap buffers.

        __syncthreads(); // Sync with the compute warps.
      }

      gC -= BN;
      gC += BDM * N;
    }
  }
  else {
    //// COMPUTE WARPS
    tId -= NUM_LOAD_THREADS;
    const unsigned int tx = tId % BDN;
    const unsigned int ty = tId / BDN;

    int mem = 0;
    float a_reg[TM][TK];
    float b_reg[TK][TN];
    float acc_reg[TM][TN];

    for(int i = 0; i < TM; i++)
      for(int j = 0; j < TN; j++)
        acc_reg[i][j] = 0.0;

    for(unsigned int bk = 0; bk < K; bk += BK) {
      __syncthreads(); // Sync with the load warps.

      for(int wk = 0; wk < BK; wk += TK) {
        // Tiled loads into Register Memory
        for(int k = 0; k < TK; k++) {
          for(int i = 0; i < TM; i++) a_reg[i][k] = sA[mem][(ty + i * BDM) * BK + (wk + k)];
          for(int j = 0; j < TN; j++) b_reg[k][j] = sB[mem][(wk + k) * BN + (tx + j * BDN)];
        }
    
        // FMA operations on Register Memory
        for(int i = 0; i < TM; i++)
          for(int j = 0; j < TN; j++)
            for(int k = 0; k < TK; k++)
              acc_reg[i][j] = fma(a_reg[i][k], b_reg[k][j], acc_reg[i][j]);
      }

      mem = 1 - mem;
    }

    for(int i = 0; i < TM; i++) {
      for(int j = 0; j < TN; j++) {
        __syncthreads(); // Sync with load warps.
        C[(bm + ty + i * BDM) * N + (bn + tx + j * BDN)] = alpha * acc_reg[i][j] + beta * sC[mem][ty * BDN + tx];
        mem = 1 - mem; // Swap buffers.
      }
    }
  }
}

#endif // DGEMM_WARP_SPECIALIZED_CUH