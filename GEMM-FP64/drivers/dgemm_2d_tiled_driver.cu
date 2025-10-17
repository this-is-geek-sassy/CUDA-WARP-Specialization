#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "drivers/dgemm_2d_tiled_driver.h" 
#include "kernels/dgemm_2d_tiled.cuh"

#define CUDA_CHECK(call)                                                          \
    ({                                                                            \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__          \
                      << " - " << cudaGetErrorString(err) << " (" #call ")" << std::endl; \
        }                                                                         \
        err == cudaSuccess; /* This is the value the macro expression returns */  \
    })

/// @brief Driver for 2D-Tiled DGEMM Kernel
/// @param M Number of rows in A
/// @param N Number of cols in B
/// @param K Number of cols in A and number of rows in B
/// @param hA Pointer to A matrix in host memory (M x K)
/// @param hB Pointer to B matrix in host memory (K x N)
/// @param hC Pointer to C matrix in host memory (M x N)
bool dgemm_2d_tiled_driver(int M, int N, int K, double* hA, double* hB, double* hC) {
  const unsigned int BM = 64;
  const unsigned int BK = 16;
  const unsigned int BN = 64;
  const unsigned int TM = 4;
  const unsigned int TN = 4;

  dim3 gridDim(N/BN, M/BM, 1);
  dim3 blockDim(BN/TN, BM/TM, 1);
  const size_t sharedMemSize = BK * (BM + BN) * sizeof(double);

  double *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if(!CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(double)))) goto cleanup;
  if(!CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(double)))) goto cleanup;
  if(!CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(double)))) goto cleanup;

  if(!CUDA_CHECK(cudaMemcpy(dA, hA, M * K * sizeof(double), cudaMemcpyHostToDevice))) goto cleanup;
  if(!CUDA_CHECK(cudaMemcpy(dB, hB, K * N * sizeof(double), cudaMemcpyHostToDevice))) goto cleanup;

  std::cout << "DRIVER: Launching 2D-Tiled Kernel..." << std::endl;
  dgemm_2d_tiled<BM, BK, BN, TM, TN><<<gridDim, blockDim, sharedMemSize>>>(M, N, K, dA, dB, dC);

  if (!CUDA_CHECK(cudaGetLastError())) goto cleanup;
  if (!CUDA_CHECK(cudaDeviceSynchronize())) goto cleanup;
  std::cout << "DRIVER: Kernel finished successfully." << std::endl;

  if(!CUDA_CHECK(cudaMemcpy(hC, dC, M * N * sizeof(double), cudaMemcpyDeviceToHost))) goto cleanup;

  cleanup:
  if(dA) cudaFree(dA);
  if(dB) cudaFree(dB);
  if(dC) cudaFree(dC);

  return cudaGetLastError() == cudaSuccess;
}