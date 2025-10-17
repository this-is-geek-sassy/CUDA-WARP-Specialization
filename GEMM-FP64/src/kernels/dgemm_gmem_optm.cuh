#ifndef DGEMM_GMEM_OPTM_CUH
#define DGEMM_GMEM_OPTM_CUH

#include <cuda.h>
#include <cassert>
#include "utils/global_mem_utils.cuh"

/// @brief Global Memory Optimized DGEMM Kernel
/// @param BM Tile Size Dimension (compile-time constant)
/// @param BK Tile Size Dimension (compile-time constant)
/// @param BN Tile Size Dimension (compile-time constant)
/// @param TM Work per thread across m-dimension (compile-time constant)
/// @param TN Work per thread across n-dimension (compile-time constant)
/// @param alpha DGEMM parameter
/// @param beta DGEMM parameter
/// @param M Number of rows in A
/// @param N Number of cols in B
/// @param K Number of cols in A and number of rows in B
/// @param A Pointer to A matrix (M x K)
/// @param B Pointer to B matrix (K x N)
/// @param C Pointer to C matrix (M x N)
template<unsigned int BM, unsigned int BK, unsigned int BN, unsigned int TM, unsigned int TN, unsigned int NUM_THREADS>
__global__ void dgemm_gmem_optm(double alpha, double beta, int M, int N, int K, double* A, double* B, double* C) {
  extern __shared__ double sm[];
  double* sA = &sm[0];
  double* sB = &sm[BM * BK];

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  unsigned int bm = blockIdx.y * BM;
  unsigned int bn = blockIdx.x * BN;

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

    for(int k = 0; k < BK; k++)
      for(int i = 0; i < TM; i++)
        for(int j = 0; j < TN; j++)
          acc_reg[i][j] = fma(sA[(ty * TM + i) * BK + k], sB[k * BN + tx * TN + j], acc_reg[i][j]);
    __syncthreads();
  }

  for(int i = 0; i < TM; i++) {
    for(int j = 0; j < TN; j++) {
      C[(bm + ty * TM + i) * N + (bn + tx * TN + j)] = acc_reg[i][j];
    }
  }
}

#endif // DGEMM_GMEM_OPTM_CUH