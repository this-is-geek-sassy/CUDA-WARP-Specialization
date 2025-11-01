#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "drivers/10_dgemm_cuda_dma_driver.h" 
#include "kernels/10_dgemm_cuda_dma.cuh"

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
bool dgemm_cuda_dma_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC) {
  const unsigned int BM = 64;
  const unsigned int BK = 16;
  const unsigned int BN = 64;
  const unsigned int TM = 4;
  const unsigned int TN = 4;
  const unsigned int TK = 2;
  const unsigned int WARP_SIZE = 32;
  const unsigned int NUM_LOAD_WARPS = 8;
  const unsigned int NUM_LOAD_THREADS = WARP_SIZE * NUM_LOAD_WARPS;
  const unsigned int BDM = BM/TM;
  const unsigned int BDN = BN/TN;
  const unsigned int NUM_COMPUTE_THREADS = BDM * BDN;

  dim3 gridDim(N/BN, M/BM, 1);
  dim3 blockDim(NUM_LOAD_THREADS + NUM_COMPUTE_THREADS, 1, 1);
  const size_t sharedMemSize = BK * (BM + BN) * 2 * sizeof(float);

  std::cout << "--- LAUNCH PARAMS ---" << std::endl;
  std::cout << "Grid:  (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << std::endl;
  std::cout << "Block: (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")" << std::endl;
  std::cout << "Threads/Block: " << (blockDim.x * blockDim.y * blockDim.z) << std::endl;
  std::cout << "Shared Mem:    " << sharedMemSize << " bytes" << std::endl;
  std::cout << "No. load threads:    " << NUM_LOAD_THREADS << std::endl;
  std::cout << "No. compute threads:    " << NUM_COMPUTE_THREADS << std::endl;
  std::cout << "---------------------" << std::endl;

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

  std::cout << "DRIVER: Launching Warp Specialized Kernel..." << std::endl;

  if(!CUDA_CHECK(cudaEventRecord(start))) goto cleanup;
  dgemm_cuda_dma<BM, BK, BN, TM, TN, TK, NUM_LOAD_THREADS, NUM_COMPUTE_THREADS, WARP_SIZE><<<gridDim, blockDim, sharedMemSize>>>(alpha, beta, M, N, K, dA, dB, dC);
  if(!CUDA_CHECK(cudaEventRecord(stop))) goto cleanup;

  if (!CUDA_CHECK(cudaGetLastError())) goto cleanup;
  if (!CUDA_CHECK(cudaDeviceSynchronize())) goto cleanup;
  std::cout << "DRIVER: Kernel finished successfully." << std::endl;

  if (!CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop))) goto cleanup;
  std::cout << "Kernel execution time: " << milliseconds * 1000 << " us" << std::endl;

  if(!CUDA_CHECK(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost))) goto cleanup;

  cleanup:
  if(dA) cudaFree(dA);
  if(dB) cudaFree(dB);
  if(dC) cudaFree(dC);

  return cudaGetLastError() == cudaSuccess;
}