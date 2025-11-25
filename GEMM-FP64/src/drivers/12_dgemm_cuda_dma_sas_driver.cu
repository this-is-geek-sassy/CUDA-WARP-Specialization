#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "drivers/12_dgemm_cuda_dma_sas_driver.h" 
#include "kernels/12_dgemm_cuda_dma_sas.cuh"

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
bool dgemm_cuda_dma_sas_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC) {
  const unsigned int TILE_SIZE = 32;
  const unsigned int COMPUTE_THREADS_PER_CTA = 256;
  const unsigned int DMA_THREADS_PER_LD = 32;
  const unsigned int NUM_DMA_LOADERS = 2;
  const unsigned int TOTAL_THREADS = COMPUTE_THREADS_PER_CTA + NUM_DMA_LOADERS * DMA_THREADS_PER_LD;

  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);
  dim3 blockDim(TOTAL_THREADS, 1, 1);

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

  std::cout << "DRIVER: Launching Bank Conflicts Free Kernel..." << std::endl;
  if(!CUDA_CHECK(cudaEventRecord(start))) goto cleanup;
  dgemm_cuda_dma_sas<
    TILE_SIZE,
    COMPUTE_THREADS_PER_CTA,
    DMA_THREADS_PER_LD,
    NUM_DMA_LOADERS,
    TOTAL_THREADS
  ><<< gridDim, blockDim >>>(M, N, K, alpha, beta, dA, dB, dC);
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