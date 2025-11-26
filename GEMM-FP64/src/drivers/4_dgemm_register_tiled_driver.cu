#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "drivers/4_dgemm_register_tiled_driver.h" 
#include "kernels/4_dgemm_register_tiled.cuh"
#include "utils/gpu_utils.cuh"
#include "headers/config.h"

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
/// @param alpha DGEMM parameter
/// @param beta DGEMM parameter
/// @param M Number of rows in A
/// @param N Number of cols in B
/// @param K Number of cols in A and number of rows in B
/// @param hA Pointer to A matrix in host memory (M x K)
/// @param hB Pointer to B matrix in host memory (K x N)
/// @param hC Pointer to C matrix in host memory (M x N)
bool dgemm_register_tiled_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC, bool debug) {
  const size_t max_optin_limit = get_max_optin_limit<0>();
  
  const unsigned int BM = BM4;
  const unsigned int BK = BK4;
  const unsigned int BN = BN4;
  const unsigned int TM = TM4;
  const unsigned int TN = TN4;
  const unsigned int TK = TK4;
  const unsigned int NUM_THREADS = (BN/TN) * (BM/TM);

  auto kernel = dgemm_register_tiled<BM, BK, BN, TM, TN, TK, NUM_THREADS>;
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kernel);
  if(MAX_SMEM) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_optin_limit);

  dim3 gridDim(N/BN, M/BM, 1);
  dim3 blockDim(BN/TN, BM/TM, 1);
  const size_t sharedMemSize = MAX_SMEM ? max_optin_limit - attr.sharedSizeBytes : BK * (BM + BN) * sizeof(float);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if(!CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)))) goto cleanup;
  if(!CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)))) goto cleanup;
  if(!CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)))) goto cleanup;

  if(!CUDA_CHECK(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice))) goto cleanup;
  if(!CUDA_CHECK(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice))) goto cleanup;
  if(!CUDA_CHECK(cudaMemcpy(dC, hC, M * N * sizeof(float), cudaMemcpyHostToDevice))) goto cleanup;

  cudaEvent_t start, stop;
  float milliseconds;
  if(!CUDA_CHECK(cudaEventCreate(&start))) goto cleanup;
  if(!CUDA_CHECK(cudaEventCreate(&stop))) goto cleanup;

  if(!CUDA_CHECK(cudaEventRecord(start))) goto cleanup;
  kernel<<<gridDim, blockDim, sharedMemSize>>>(alpha, beta, M, N, K, dA, dB, dC);
  if(!CUDA_CHECK(cudaEventRecord(stop))) goto cleanup;

  if (!CUDA_CHECK(cudaGetLastError())) goto cleanup;
  if (!CUDA_CHECK(cudaDeviceSynchronize())) goto cleanup;

  if (!CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop))) goto cleanup;
  std::cout << milliseconds * 1000 << std::endl;

  if(!CUDA_CHECK(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost))) goto cleanup;

  cleanup:
  if(dA) cudaFree(dA);
  if(dB) cudaFree(dB);
  if(dC) cudaFree(dC);

  return cudaGetLastError() == cudaSuccess;
}