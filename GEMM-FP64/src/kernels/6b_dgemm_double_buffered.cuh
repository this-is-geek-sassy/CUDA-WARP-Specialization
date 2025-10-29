#ifndef DGEMM_DOUBLE_BUFFERED_CUH
#define DGEMM_DOUBLE_BUFFERED_CUH

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
template<unsigned int BM, unsigned int BK, unsigned int BN, unsigned int TM, unsigned int TN, unsigned int TK, unsigned int NUM_THREADS>
__global__ void dgemm_double_buffered(float alpha, float beta, int M, int N, int K, float* A, float* B, float* C) {
  extern __shared__ float sm[];
  float* sA[2] = {&sm[0], &sm[BM * BK]};
  float* sB[2] = {&sm[2 * BM * BK], &sm[2 * BM * BK + BK * BN]};
  int mem = 0, buf = 1;

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  constexpr unsigned int BDM = (BM/TM); // blockIdx.y (compile time constant)
  constexpr unsigned int BDN = (BN/TN); // blockIdx.x (compile time constant)

  unsigned int bm = blockIdx.y * BM;
  unsigned int bn = blockIdx.x * BN;

  float a_reg[TM][TK];
  float b_reg[TK][TN];
  float acc_reg[TM][TN];
  for(int i = 0; i < TM; i++)
    for(int j = 0; j < TN; j++)
      acc_reg[i][j] = 0.0;

  // Pre-load the first tile.
  float* gA = A + bm * K;
  float* gB = B + bn;
  readTileChunked<BM, BK, NUM_THREADS>(K, gA, sA[mem]);
  readTileChunked<BK, BN, NUM_THREADS>(N, gB, sB[mem]);
    __syncthreads();

  // Update pointers to the next tile.
  gA += BK;
  gB += BK * N;
  
  for(unsigned int bk = BK; bk < K; bk += BK) {
    // Load the next tile into extra buffers.
    readTileChunked<BM, BK, NUM_THREADS>(K, gA, sA[buf]);
    readTileChunked<BK, BN, NUM_THREADS>(N, gB, sB[buf]);

    // Process the current tile.
    for(int wk = 0; wk < BK; wk += TK) {
      // Tiled loads into Register Memory (Need to check PTX and SASS to confirm unrolling and chunking)
      for(int k = 0; k < TK; k++) {
        for(int i = 0; i < TM; i++) a_reg[i][k] = sA[mem][(ty + i * BDM) * BK + (wk + k)];
        for(int j = 0; j < TN; j++) b_reg[k][j] = sB[mem][(wk + k) * BN + (tx + j * BDN)];
      }
  
      // FMA operations on Register Memory (Need to check PTX and SASS to confirm unrolling)
      for(int i = 0; i < TM; i++)
        for(int j = 0; j < TN; j++)
          for(int k = 0; k < TK; k++)
            acc_reg[i][j] = fma(a_reg[i][k], b_reg[k][j], acc_reg[i][j]);
    }
    __syncthreads();

    // Update pointers to next tile.
    gA += BK;
    gB += BK * N;

    // Swap buffers.
    buf = 1 - buf;
    mem = 1 - mem;
  }

  // Process the last tile.
  for(int wk = 0; wk < BK; wk += TK) {
    // Tiled loads into Register Memory (Need to check PTX and SASS to confirm unrolling and chunking)
    for(int k = 0; k < TK; k++) {
      for(int i = 0; i < TM; i++) a_reg[i][k] = sA[mem][(ty + i * BDM) * BK + (wk + k)];
      for(int j = 0; j < TN; j++) b_reg[k][j] = sB[mem][(wk + k) * BN + (tx + j * BDN)];
    }

    // FMA operations on Register Memory (Need to check PTX and SASS to confirm unrolling)
    for(int i = 0; i < TM; i++)
      for(int j = 0; j < TN; j++)
        for(int k = 0; k < TK; k++)
          acc_reg[i][j] = fma(a_reg[i][k], b_reg[k][j], acc_reg[i][j]);
  }

  for(int i = 0; i < TM; i++) {
    for(int j = 0; j < TN; j++) {
      C[(bm + ty + i * BDM) * N + (bn + tx + j * BDN)] = alpha * acc_reg[i][j] + beta * C[(bm + ty + i * BDM) * N + (bn + tx + j * BDN)];
    }
  }
}

#endif // DGEMM_DOUBLE_BUFFERED_CUH