#ifndef DGEMM_CUDA_DMA_CUH
#define DGEMM_CUDA_DMA_CUH

#include <cuda.h>
#include <cassert>
#include "../headers/cudaDMAv2.h"
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
         unsigned int NUM_LOAD_WARPS, unsigned int NUM_COMPUTE_WARPS, unsigned int WARP_SIZE>
__global__ void dgemm_cuda_dma(float alpha, float beta, int M, int N, int K, float* A, float* B, float* C) {
  constexpr unsigned int BDM = (BM/TM); // blockDim.y (compile time constant)
  constexpr unsigned int BDN = (BN/TN); // blockDim.x (compile time constant)

  constexpr unsigned int NUM_LOAD_THREADS_PER_LD = NUM_LOAD_WARPS * WARP_SIZE / 2;
  constexpr unsigned int NUM_COMPUTE_THREADS = NUM_COMPUTE_WARPS * WARP_SIZE;

  extern __shared__ float sm[];
  float* sA[2] = {&sm[0], &sm[BM * BK]};
  float* sB[2] = {&sm[2 * BM * BK], &sm[2 * BM * BK + BK * BN]};

  const unsigned int bm = blockIdx.y * BM;
  const unsigned int bn = blockIdx.x * BN;

  const unsigned int tId = threadIdx.y * blockDim.x + threadIdx.x;

  CudaDMAStrided<true, 16, 128, sizeof(float)*BK, NUM_LOAD_THREADS_PER_LD, BM>
    dma_A(1, NUM_COMPUTE_THREADS, NUM_COMPUTE_THREADS, sizeof(float)*K, sizeof(float)*BK);
  CudaDMAStrided<true, 16, 128, sizeof(float)*BN, NUM_LOAD_THREADS_PER_LD, BK>
    dma_B(2, NUM_COMPUTE_THREADS, NUM_COMPUTE_THREADS + NUM_LOAD_THREADS_PER_LD, sizeof(float)*N, sizeof(float)*BN);

  if(tId < NUM_COMPUTE_THREADS) {
    //// COMPUTE WARPS
    const unsigned int tx = tId % BDN;
    const unsigned int ty = tId / BDN;

    int mem = 0;
    float acc_reg[TM][TN];

    for(int i = 0; i < TM; i++)
      for(int j = 0; j < TN; j++)
        acc_reg[i][j] = 0.0;

    // Start load of the first tile.
    dma_A.start_async_dma();
    dma_B.start_async_dma();

    for(unsigned int bk = BK; bk < K; bk += BK) {
      dma_A.wait_for_dma_finish();
      dma_B.wait_for_dma_finish();

      for(int k = 0; k < BK; k++)
        for(int i = 0; i < TM; i++)
          for(int j = 0; j < TN; j++)
            acc_reg[i][j] = fma(sA[mem][(ty + i * BDM) * BK + k], sB[mem][k * BN + (tx + j * BDN)], acc_reg[i][j]);

      mem ^= 1;

      dma_A.start_async_dma();
      dma_B.start_async_dma();
    }
    
    dma_A.wait_for_dma_finish();
    dma_B.wait_for_dma_finish();

    for(int k = 0; k < BK; k++)
      for(int i = 0; i < TM; i++)
        for(int j = 0; j < TN; j++)
          acc_reg[i][j] = fma(sA[mem][(ty + i * BDM) * BK + k], sB[mem][k * BN + (tx + j * BDN)], acc_reg[i][j]);

    // Epilogue
    for(int i = 0; i < TM; i++) {
      for(int j = 0; j < TN; j++) {
        C[(bm + ty + i * BDM) * N + (bn + tx + j * BDN)] = alpha * acc_reg[i][j] + beta * C[(bm + ty + i * BDM) * N + (bn + tx + j * BDN)];
      }
    }
  } else if(dma_A.owns_this_thread()) {
    int buf = 0;
    float* gA = A + bm * K;
    for(unsigned int bk = 0; bk < K; bk += BK) {
      dma_A.execute_dma(gA, sA[buf]);
      gA += BK;
      buf ^= 1;
    }
  } else if(dma_B.owns_this_thread()) {
    int buf = 0;
    float* gB = B + bn;
    for(unsigned int bk = 0; bk < K; bk += BK) {
      dma_B.execute_dma(gB, sB[buf]);
      gB += BK * N;
      buf ^= 1;
    }
  }
}

#endif // DGEMM_CUDA_DMA_CUH