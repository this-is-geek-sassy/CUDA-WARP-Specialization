#ifndef DGEMM_REGISTER_TILED_CUH
#define DGEMM_REGISTER_TILED_CUH

#include <cuda.h>
#include <cassert>
#include "utils/global_mem_utils.cuh"

/// @brief Register Tiled DGEMM Kernel
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
__global__ void dgemm_register_tiled(double alpha, double beta, int M, int N, int K, double* A, double* B, double* C) {
  extern __shared__ double sm[];
  double* sA = &sm[0];
  double* sB = &sm[BM * BK];

  constexpr unsigned int TM_STEP = BM / TM;
  constexpr unsigned int TN_STEP = BN / TN;

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  unsigned int bm = blockIdx.y * BM;
  unsigned int bn = blockIdx.x * BN;

  double a_reg[TM][TK];
  double b_reg[TK][TN];
  double acc_reg[TM][TN];
  for(int i = 0; i < TM; i++) 
    for(int j = 0; j < TN; j++)
      acc_reg[i][j] = 0.0;

  for(unsigned int bk = 0; bk < K; bk += BK) {
    double* gA = A + (bm * K + bk);
    double* gB = B + (bk * N + bn);
    readTileChunked<BM, BK, NUM_THREADS>(K, gA, sA);
    readTileChunked<BK, BN, NUM_THREADS>(N, gB, sB);
    __syncthreads();

    for(int wk = 0; wk < BK; wk += TK) {
      // Tiled loads into Register Memory (Need to check PTX and SASS to confirm unrolling and chunking)
      for(int k = 0; k < TK; k++) {
        for(int i = 0; i < TM; i++) a_reg[i][k] = sA[(ty + i * TM_STEP) * BK + wk + k];
        for(int j = 0; j < TN; j++) b_reg[k][j] = sB[(wk + k) * BN + tx + j * TN_STEP]; 
      }
  
      // FMA operations on Register Memory (Need to check PTX and SASS to confirm unrolling)
      for(int i = 0; i < TM; i++)
        for(int j = 0; j < TN; j++)
          for(int k = 0; k < TK; k++)
            acc_reg[i][j] = fma(a_reg[i][k], b_reg[k][j], acc_reg[i][j]);
    }
    __syncthreads();
  }

  for(int i = 0; i < TM; i++) 
    for(int j = 0; j < TN; j++) 
      C[(bm + ty + i * TM_STEP) * N + (bn + tx + j * TN_STEP)] = alpha * acc_reg[i][j] + beta * C[(bm + ty + i * TM_STEP) * N + (bn + tx + j * TN_STEP)]; 
}

#endif // DGEMM_REGISTER_TILED_CUH