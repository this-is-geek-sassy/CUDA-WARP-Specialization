#ifndef DGEMM_2D_TILED_CUH
#define DGEMM_2D_TILED_CUH

#include <cuda.h>
#include <cassert>

/// @brief Loads a 2D-Tile from src into dest
/// @param BM Tile Size Dimension (compile-time constant)
/// @param BN Tile Size Dimension (compile-time constant)
/// @param N Row stride of src
/// @param src Pointer to src
/// @param dest Pointer to dest
template<unsigned int BM, unsigned int BN>
__device__ void loadTile(const unsigned int N, double* src, double* dest) {
  const unsigned int NUM_THREADS = blockDim.x * blockDim.y;

  assert(NUM_THREADS % BN == 0);
  const unsigned int ROW_STEP = NUM_THREADS / BN; 

  const unsigned int tId = threadIdx.y * blockDim.y + threadIdx.x;
  const unsigned int row = tId / BN;
  const unsigned int col = tId % BN;

  for(unsigned int i = row; i < BM; i += ROW_STEP) {
    dest[i * BN + col] = src[i * N + col];
  }
}

/// @brief 2D-Tiled DGEMM Kernel
/// @param BM Tile Size Dimension (compile-time constant)
/// @param BK Tile Size Dimension (compile-time constant)
/// @param BN Tile Size Dimension (compile-time constant)
/// @param TM Work per thread across m-dimension (compile-time constant)
/// @param TN Work per thread across n-dimension (compile-time constant)
/// @param M Number of rows in A
/// @param N Number of cols in B
/// @param K Number of cols in A and number of rows in B
/// @param A Pointer to A matrix (M x K)
/// @param B Pointer to B matrix (K x N)
/// @param C Pointer to C matrix (M x N)
template<unsigned int BM, unsigned int BK, unsigned int BN, unsigned int TM, unsigned int TN>
__global__ void dgemm_2d_tiled(int M, int N, int K, double* A, double* B, double* C) {
  extern __shared__ double sm[];
  double* sA = &sm[0];
  double* sB = &sm[BM * BK];

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  unsigned int bm = blockIdx.y * BM + tx;
  unsigned int bn = blockIdx.x * BN + ty;

  double acc_reg[TM][TN];
  for(unsigned int bk = 0; bk < K; bk += BK) {
    double* gA = A + (bm * K + bk);
    double* gB = B + (bk * N + bn);
    loadTile<BM, BK>(K, gA, sA);
    loadTile<BK, BN>(N, gB, sB);
    __syncthreads();

    for(int k = 0; k < BK; k++)
      for(int i = 0; i < TM; i++)
        for(int j = 0; j < TN; j++)
          acc_reg[i][j] += sA[(ty * TM + i) * BK + k] * sB[k * BN + tx * TN + j];
    __syncthreads();
  }

  for(int i = 0; i < TM; i++) {
    for(int j = 0; j < TN; j++) {
      C[(bm + ty * TM + i) * N + (bn + tx * TN + j)] = acc_reg[i][j];
    }
  }
}

#endif // DGEMM_2D_TILED_CUH