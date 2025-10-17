#ifndef DGEMM_OVERLAPPED_CUH
#define DGEMM_OVERLAPPED_CUH

#include <cuda.h>
#include <cassert>
#include "utils/global_mem_utils.cuh"

/// @brief Computation Overlapped Global Memory Reads DGEMM Kernel
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
__global__ void dgemm_overlapped(double alpha, double beta, int M, int N, int K, double* A, double* B, double* C) {
  extern __shared__ double sm[];
  double* sA = &sm[0];
  double* sB = &sm[BM * BK];

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  constexpr unsigned int BDM = (BM/TM); // blockIdx.y (compile time constant)
  constexpr unsigned int BDN = (BN/TN); // blockIdx.x (compile time constant)

  unsigned int bm = blockIdx.y * BM;
  unsigned int bn = blockIdx.x * BN;

  double a_reg[TM][TK];
  double b_reg[TK][TN];
  double acc_reg[TM][TN];
  for(int i = 0; i < TM; i++)
    for(int j = 0; j < TN; j++)
      acc_reg[i][j] = 0.0;

  // Allocate registers for loading next iteration's tile.
  constexpr unsigned int BN_VECTORIZED = BN / 2;
  constexpr unsigned int ROW_STEP = NUM_THREADS / BN_VECTORIZED; 
  constexpr unsigned int NUM_ITERS = BM / ROW_STEP;
  float4 sA_reg[NUM_ITERS];
  float4 sB_reg[NUM_ITERS];

  // Load first tile.
  double* gA = A + (bm * K);
  double* gB = B + bn;
  readTileChunked<BM, BK, NUM_THREADS>(K, gA, sA);
  readTileChunked<BK, BN, NUM_THREADS>(N, gB, sB);
  __syncthreads();

  // Update the global memory indexes to next tile.
  gA += BK;
  gB += BK * N;

  for(unsigned int bk = BK; bk < K; bk += BK) {
    // Load the next tile from global memory into registers.
    loadTileChunked<BM, BK, NUM_THREADS, NUM_ITERS>(K, gA, sA_reg);
    loadTileChunked<BK, BN, NUM_THREADS, NUM_ITERS>(N, gB, sB_reg);

    // Perform computation on current tile.
    for(int wk = 0; wk < BK; wk += TK) {
      for(int k = 0; k < TK; k++) {
        for(int i = 0; i < TM; i++) a_reg[i][k] = sA[(ty + i * BDM) * BK + (wk + k)];
        for(int j = 0; j < TN; j++) b_reg[k][j] = sB[(wk + k) * BN + (tx + j * BDN)];
      }
  
      for(int i = 0; i < TM; i++)
        for(int j = 0; j < TN; j++)
          for(int k = 0; k < TK; k++)
            acc_reg[i][j] = fma(a_reg[i][k], b_reg[k][j], acc_reg[i][j]);
    }
    __syncthreads();

    // Update the global memory indexes to next tile.
    gA += BK;
    gB += BK * N;

    // Store the next tile from registers into shared memory.
    storeTileChunked<BM, BK, NUM_THREADS, NUM_ITERS>(sA_reg, sA);
    storeTileChunked<BK, BN, NUM_THREADS, NUM_ITERS>(sB_reg, sB);
    __syncthreads();
  }

  // Perform computation on last tile.
  for(int wk = 0; wk < BK; wk += TK) {
    for(int k = 0; k < TK; k++) {
      for(int i = 0; i < TM; i++) a_reg[i][k] = sA[(ty + i * BDM) * BK + (wk + k)];
      for(int j = 0; j < TN; j++) b_reg[k][j] = sB[(wk + k) * BN + (tx + j * BDN)];
    }

    for(int i = 0; i < TM; i++)
      for(int j = 0; j < TN; j++)
        for(int k = 0; k < TK; k++)
          acc_reg[i][j] = fma(a_reg[i][k], b_reg[k][j], acc_reg[i][j]);
  }
  __syncthreads();

  for(int i = 0; i < TM; i++) {
    for(int j = 0; j < TN; j++) {
      C[(bm + ty + i * BDM) * N + (bn + tx + j * BDN)] = alpha * acc_reg[i][j] + beta * C[(bm + ty + i * BDM) * N + (bn + tx + j * BDN)];
    }
  }
}

#endif // DGEMM_OVERLAPPED_CUH