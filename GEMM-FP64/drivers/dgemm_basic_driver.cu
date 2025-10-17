#include <cuda.h>
#include "drivers/dgemm_basic_driver.h" 
#include "kernels/dgemm_basic.cuh"

/// @brief Driver for Basic DGEMM Kernel
/// @param M Number of rows in A
/// @param N Number of cols in B
/// @param K Number of cols in A and number of rows in B
/// @param hA Pointer to A matrix in host memory (M x K)
/// @param hB Pointer to B matrix in host memory (K x N)
/// @param hC Pointer to C matrix in host memory (M x N)
void dgemm_basic_driver(int M, int N, int K, double* hA, double* hB, double* hC) {
  double *dA, *dB, *dC;
  cudaMalloc(&dA, M * K * sizeof(double));
  cudaMalloc(&dB, K * N * sizeof(double));
  cudaMalloc(&dC, M * N * sizeof(double));
  cudaMemcpy(dA, hA, M * K * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, K * N * sizeof(double), cudaMemcpyHostToDevice);

  const unsigned int TS = 32;
  const size_t sharedMemSize = 2 * TS * TS * sizeof(double);
  dim3 gridDim(N/TS, M/TS, 1);
  dim3 blockDim(TS, TS, 1);

  dgemm_basic<TS><<<gridDim, blockDim, sharedMemSize>>>(M, N, K, dA, dB, dC);

  cudaMemcpy(hC, dC, M * N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}