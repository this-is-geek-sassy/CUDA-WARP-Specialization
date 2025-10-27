#ifndef DGEMM_BASIC_CUH
#define DGEMM_BASIC_CUH

#include <cuda.h>

/// @brief A basic DGEMM Kernel
/// @param TS Tile Size (compile-time constant)
/// @param alpha DGEMM parameter
/// @param beta DGEMM parameter
/// @param M Number of rows in A
/// @param N Number of cols in B
/// @param K Number of cols in A and number of rows in B
/// @param A Pointer to A matrix (M x K)
/// @param B Pointer to B matrix (K x N)
/// @param C Pointer to C matrix (M x N)
template<unsigned int TS>
__global__ void dgemm_basic(float alpha, float beta, int M, int N, int K, float* A, float* B, float* C) {
  extern __shared__ float sm[];
  float* sA = &sm[0];
  float* sB = &sm[TS * TS];

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  unsigned int bm = blockIdx.y * TS;
  unsigned int bn = blockIdx.x * TS;

  float acc_reg = 0.0;
  for(unsigned int bk = 0; bk < K; bk += TS) {
    sA[ty * TS + tx] = A[(bm + ty) * K + (bk + tx)];
    sB[ty * TS + tx] = B[(bk + ty) * N + (bn + tx)];
    __syncthreads();

    for(int k = 0; k < TS; k++)
      acc_reg += sA[ty * TS + k] * sB[k * TS + tx];
    __syncthreads();
  }

  C[(bm + ty) * N + (bn + tx)] = alpha * acc_reg + beta * C[(bm + ty) * N + (bn + tx)];
}

#endif // DGEMM_BASIC_CUH