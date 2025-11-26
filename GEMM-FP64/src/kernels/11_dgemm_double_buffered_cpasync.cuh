#ifndef DGEMM_DOUBLE_BUFFERED_CPASYNC_CUH
#define DGEMM_DOUBLE_BUFFERED_CPASYNC_CUH

#include <cuda.h>
#include <cassert>
#include <cuda/barrier>
#include "utils/global_mem_utils.cuh"

/// @brief Double Buffered DGEMM Kernel
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
         unsigned int NUM_THREADS>
__global__ void dgemm_double_buffered_cpasync(float alpha, float beta, int M, int N, int K, float* A, float* B, float* C) {
  constexpr unsigned int BDN = (BN/TN); // blockDim.x (compile time constant)

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ float sm[];

  float* sA[2] = {&sm[0], &sm[BM * BK]};
  float* sB[2] = {&sm[2 * BM * BK], &sm[2 * BM * BK + BK * BN]};
  int mem = 0, buf = 1;

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;
  const unsigned int tId = ty * BDN + tx;

  unsigned int bm = blockIdx.y * BM;
  unsigned int bn = blockIdx.x * BN;

  float acc_reg[TM][TN];
  for(int i = 0; i < TM; i++)
    for(int j = 0; j < TN; j++)
      acc_reg[i][j] = 0.0;

  if (tId == 0) {
    init(&bar, NUM_THREADS);
  }
  __syncthreads();

  // Pre-load the first tile.
  if(tId < BM) 
    cuda::memcpy_async(
      &sA[mem][tId * BK],
      &A[(bm + tId) * K],
      cuda::aligned_size_t<16>(BK * sizeof(float)),
      bar
    );

  if(tId < BK)
    cuda::memcpy_async(
      &sB[mem][tId * BN],
      &B[tId * N + bn],
      cuda::aligned_size_t<16>(BN * sizeof(float)),
      bar
    );

  // Wait for the loads to complete.
  bar.arrive_and_wait();

  for(unsigned int bk = BK; bk < K; bk += BK) {
    // Load the next tile.
    if(tId < BM) 
      cuda::memcpy_async(
        &sA[buf][tId * BK],
        &A[(bm + tId) * K + bk],
        cuda::aligned_size_t<16>(BK * sizeof(float)),
        bar
      );

    if(tId < BK)
      cuda::memcpy_async(
        &sB[buf][tId * BN],
        &B[(tId + bk) * N + bn],
        cuda::aligned_size_t<16>(BN * sizeof(float)),
        bar
      );

    // Process the current tile.
      for(int k = 0; k < BK; k++)
        for(int i = 0; i < TM; i++)
          for(int j = 0; j < TN; j++)
            acc_reg[i][j] = fma(sA[mem][(ty * TM + i) * BK + k], sB[mem][k * BN + tx * TN + j], acc_reg[i][j]);

    // Wait for the next tile to be loaded.
    bar.arrive_and_wait();

    // Swap buffers.
    buf ^= 1;
    mem ^= 1;
  } 

  // Process the last tile.
  for(int k = 0; k < BK; k++)
    for(int i = 0; i < TM; i++)
      for(int j = 0; j < TN; j++)
        acc_reg[i][j] = fma(sA[mem][(ty * TM + i) * BK + k], sB[mem][k * BN + tx * TN + j], acc_reg[i][j]);
 
  // Epilogue
  for(int i = 0; i < TM; i++)
    for(int j = 0; j < TN; j++)
      C[(bm + ty * TM + i) * N + (bn + tx * TN + j)] = alpha * acc_reg[i][j] + beta * C[(bm + ty * TM + i) * N + (bn + tx * TN + j)];

}

#endif // DGEMM_DOUBLE_BUFFERED_CPASYNC_CUH