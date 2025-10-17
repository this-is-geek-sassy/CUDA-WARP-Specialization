#ifndef DGEMM_BASIC_CUH
#define DGEMM_BASIC_CUH

#include <cuda.h>

/// @brief A basic DGEMM Kernel
/// @param TS Tile Size (compile-time constant)
/// @param M Number of rows in A
/// @param N Number of cols in B
/// @param K Number of cols in A and number of rows in B
/// @param A Pointer to A matrix (M x K)
/// @param B Pointer to B matrix (K x N)
/// @param C Pointer to C matrix (M x N)
template<unsigned int TS>
__global__ void dgemm_basic(int M, int N, int K, double* A, double* B, double* C) {
  extern __shared__ double sm[];
  double* sA = &sm[0];
  double* sB = &sm[TS * TS];

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  unsigned int bm = blockIdx.y * TS;
  unsigned int bn = blockIdx.x * TS;

  double acc_reg = 0.0;
  for(unsigned int bk = 0; bk < K; bk += TS) {
    sA[ty * TS + tx] = A[(bm + ty) * K + (bk + tx)];
    sB[ty * TS + tx] = B[(bk + ty) * N + (bn + tx)];
    __syncthreads();

    for(int k = 0; k < TS; k++)
      acc_reg += sA[ty * TS + k] * sB[k * TS + tx];
    __syncthreads();
  }

  C[(bm + ty) * N + (bn + tx)] = acc_reg;
}

#endif // DGEMM_BASIC_CUH