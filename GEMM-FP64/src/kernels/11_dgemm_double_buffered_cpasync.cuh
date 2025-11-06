#ifndef DGEMM_DOUBLE_BUFFERED_CPASYNC_CUH
#define DGEMM_DOUBLE_BUFFERED_CPASYNC_CUH

#include <cuda.h>
#include <cassert>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
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
template<unsigned int BM, unsigned int BK, unsigned int BN, unsigned int TM, unsigned int TN, unsigned int TK, unsigned int NUM_THREADS>
__global__ void dgemm_double_buffered_cpasync(float alpha, float beta, int M, int N, int K, float* A, float* B, float* C) {
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();
  auto tileA = cg::tiled_partition<BM>(block);
  auto tileB = cg::tiled_partition<BK>(block);

  constexpr unsigned int BDM = (BM/TM); // blockDim.y (compile time constant)
  constexpr unsigned int BDN = (BN/TN); // blockDim.x (compile time constant)

  __shared__ float sm[2 * BK * (BM + BN)];
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


  // Pre-load the first tile.
  cg::memcpy_async(tileA, &sA[mem][tileA.meta_group_rank() * BK], &A[(bm + tileA.meta_group_rank()) * K], cuda::aligned_size_t<16>(BK * sizeof(float)));
  cg::memcpy_async(tileB, &sB[mem][tileB.meta_group_rank() * BN], &B[tileB.meta_group_rank() * N + bn], cuda::aligned_size_t<16>(BN * sizeof(float)));

  // Wait for the loads to complete.
  cg::wait(block);
  
  for(unsigned int bk = BK; bk < K; bk += BK) {
    cg::memcpy_async(tileA, &sA[buf][tileA.meta_group_rank() * BK], &A[(bm + tileA.meta_group_rank()) * K + bk], cuda::aligned_size_t<16>(BK * sizeof(float)));
    cg::memcpy_async(tileB, &sB[buf][tileB.meta_group_rank() * BN], &B[(tileB.meta_group_rank() + bk) * N + bn], cuda::aligned_size_t<16>(BN * sizeof(float)));

    // Process the current tile.
    for(int k = 0; k < BK; k++)
      for(int i = 0; i < TM; i++)
        for(int j = 0; j < TN; j++)
          acc_reg[i][j] = fma(sA[mem][(ty + i * BDM) * BK + k], sB[mem][k * BN + (tx + j * BDN)], acc_reg[i][j]);

    
    // Wait for the next tile to be loaded.
    cg::wait(block);

    // Swap buffers.
    buf ^= 1;
    mem ^= 1;
  }

  // Process the last tile.
  for(int k = 0; k < BK; k++)
    for(int i = 0; i < TM; i++)
      for(int j = 0; j < TN; j++)
        acc_reg[i][j] = fma(sA[mem][(ty + i * BDM) * BK + k], sB[mem][k * BN + (tx + j * BDN)], acc_reg[i][j]);
 
  // Epilogue
  for(int i = 0; i < TM; i++)
    for(int j = 0; j < TN; j++)
      C[(bm + ty + i * BDM) * N + (bn + tx + j * BDN)] = alpha * acc_reg[i][j] + beta * C[(bm + ty + i * BDM) * N + (bn + tx + j * BDN)];
}

#endif // DGEMM_DOUBLE_BUFFERED_CPASYNC_CUH