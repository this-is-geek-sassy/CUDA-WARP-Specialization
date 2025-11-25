#ifndef CUDA_DMA_SAS_DRIVER_H
#define CUDA_DMA_SAS_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

bool dgemm_cuda_dma_sas_driver(float alpha, float beta, int M, int N, int K, float* hA, float* hB, float* hC);

#ifdef __cplusplus
}
#endif

#endif // CUDA_DMA_SAS_DRIVER_H