#ifndef DGEMM_WARP_SPECIALIZED_CPASYNC_CUH
#define DGEMM_WARP_SPECIALIZED_CPASYNC_CUH

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
         unsigned int NUM_THREADS, unsigned int NUM_LOAD_WARPS, unsigned int WARP_SIZE>
__global__ void dgemm_warp_specialized_cpasync(float alpha, float beta, int M, int N, int K, float* A, float* B, float* C) {
  constexpr unsigned int BDN = (BN/TN); // blockDim.x (compile time constant)
  constexpr unsigned int NUM_LOAD_THREADS = NUM_LOAD_WARPS * WARP_SIZE;

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ float sm[];

  float* sA[2] = {&sm[0], &sm[BM * BK]};
  float* sB[2] = {&sm[2 * BM * BK], &sm[2 * BM * BK + BK * BN]};

  const unsigned int bm = blockIdx.y * BM;
  const unsigned int bn = blockIdx.x * BN;

  unsigned int tId = threadIdx.y * blockDim.x + threadIdx.x;
  const bool isLoadWarp = tId < NUM_LOAD_THREADS;

  if (tId == 0) {
    init(&bar, NUM_THREADS);
  }
  __syncthreads();

  if(isLoadWarp) {
    //// LOAD WARPS
    int buf = 0;
    float* gA = A + bm * K;
    float* gB = B + bn;

    for(unsigned int bk = 0; bk < K; bk += BK) {
      // Load the next tile into sA.
      constexpr unsigned int NUM_SA_ITERS = (BM + NUM_LOAD_THREADS - 1) / NUM_LOAD_THREADS;
      unsigned int row = tId;

      for(int i = 0; i < NUM_SA_ITERS; i++) {
        if(row < BM)
          cuda::memcpy_async(
            &sA[buf][row * BK],
            &gA[row * K],
            cuda::aligned_size_t<16>(BK * sizeof(float)),
            bar
          );
        row += NUM_LOAD_THREADS;
      }

      // Load the next tile into sB.
      constexpr unsigned int NUM_SB_ITERS = (BK + NUM_LOAD_THREADS - 1) / NUM_LOAD_THREADS;
      row = tId;

      for(int i = 0; i < NUM_SB_ITERS; i++) {
        if(row < BK)
          cuda::memcpy_async(
            &sB[buf][row * BN],
            &gB[row * N],
            cuda::aligned_size_t<16>(BN * sizeof(float)),
            bar
          );
        row += NUM_LOAD_THREADS;
      }
    
      // Update pointers to next tile.
      gA += BK;
      gB += BK * N;

      // Swap buffers.
      buf ^= 1;

      // Sync with the compute threads.
      bar.arrive_and_wait();
    }
  } else {
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
      bar.arrive_and_wait(); // Sync with the load warps.

      for(int k = 0; k < BK; k++)
        for(int i = 0; i < TM; i++)
          for(int j = 0; j < TN; j++)
            acc_reg[i][j] = fma(sA[mem][(ty * TM + i) * BK + k], sB[mem][k * BN + tx * TN + j], acc_reg[i][j]);

      mem ^= 1; // Swap Buffers
    }

    // Epilogue
    for(int i = 0; i < TM; i++)
      for(int j = 0; j < TN; j++)
        C[(bm + ty * TM + i) * N + (bn + tx * TN + j)] = alpha * acc_reg[i][j] + beta * C[(bm + ty * TM + i) * N + (bn + tx * TN + j)];
  }
}

#endif // DGEMM_WARP_SPECIALIZED_CPASYNC_CUH