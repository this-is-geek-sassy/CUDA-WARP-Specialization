#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define POLYBENCH_TIME 1

#include "gemm_fp32.cuh"
#include "common/polybench.h"
#include "common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Declared constant values for ALPHA and BETA */
#define ALPHA 32412.0f
#define BETA 2123.0f

#define RUN_ON_CPU

void gemm(int ni, int nj, int nk, fp32_t alpha, fp32_t beta, fp32_t POLYBENCH_2D(A,NI,NK,ni,nk), 
         fp32_t POLYBENCH_2D(B,NK,NJ,nk,nj), fp32_t POLYBENCH_2D(C,NI,NJ,ni,nj))
{
    int i, j, k;
    
    for (i = 0; i < _PB_NI; i++)
    {
        for (j = 0; j < _PB_NJ; j++)
        {
            C[i][j] *= beta;
            for (k = 0; k < _PB_NK; ++k)
            {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
}

void init(int ni, int nj, int nk, fp32_t* alpha, fp32_t* beta, fp32_t POLYBENCH_2D(A,NI,NK,ni,nk), 
        fp32_t POLYBENCH_2D(B,NK,NJ,nk,nj), fp32_t POLYBENCH_2D(C,NI,NJ,ni,nj))
{
    int i, j;

    *alpha = 32412.0f;
    *beta = 2123.0f;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nk; j++)
        {
            A[i][j] = ((fp32_t) i*j) / NI;
        }
    }

    for (i = 0; i < nk; i++)
    {
        for (j = 0; j < nj; j++)
        {
            B[i][j] = ((fp32_t) i*j) / NI;
        }
    }

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            C[i][j] = ((fp32_t) i*j) / NI;
        }
    }
}

void compareResults(int ni, int nj, fp32_t POLYBENCH_2D(C,NI,NJ,ni,nj), 
                   fp32_t POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
    int i, j, fail;
    fail = 0;
    
    // Debug: Print first few mismatches
    int printCount = 0;
    
    for (i=0; i < ni; i++) 
    {
        for (j=0; j < nj; j++) 
        {
            fp32_t diff = percentDiff(C[i][j], C_outputFromGpu[i][j]);
            if (diff > PERCENT_DIFF_ERROR_THRESHOLD) 
            {
                fail++;
                if (printCount < 5) {
                    printf("Mismatch at [%d][%d]: CPU=%.6e, GPU=%.6e, diff=%.2f%%\n", 
                           i, j, C[i][j], C_outputFromGpu[i][j], diff);
                    printCount++;
                }
            }
        }
    }
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", 
           PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("Setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

__global__ void gemm_kernel_fp32(int ni, int nj, int nk, fp32_t alpha, fp32_t beta, 
                                fp32_t *a, fp32_t *b, fp32_t *c)
{
    // Shared memory for tiling
    __shared__ fp32_t As[TILE_SIZE][TILE_SIZE];
    __shared__ fp32_t Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global row and column for this thread
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    fp32_t sum = 0.0f;
    
    // Loop over tiles
    int numTiles = (nk + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile from matrix A into shared memory
        int aCol = t * TILE_SIZE + tx;
        if (row < ni && aCol < nk)
            As[ty][tx] = alpha * a[row * nk + aCol];
        else
            As[ty][tx] = 0.0f;
        
        // Load tile from matrix B into shared memory
        int bRow = t * TILE_SIZE + ty;
        if (bRow < nk && col < nj)
            Bs[ty][tx] = b[bRow * nj + col];
        else
            Bs[ty][tx] = 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < ni && col < nj) {
        c[row * nj + col] = beta * c[row * nj + col] + sum;
    }
}

void gemmCuda_fp32(int ni, int nj, int nk, fp32_t alpha, fp32_t beta, 
                   fp32_t POLYBENCH_2D(A,NI,NK,ni,nk), 
                   fp32_t POLYBENCH_2D(B,NK,NJ,nk,nj), 
                   fp32_t POLYBENCH_2D(C,NI,NJ,ni,nj), 
                   fp32_t POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
    fp32_t *A_gpu;
    fp32_t *B_gpu;
    fp32_t *C_gpu;

    cudaMalloc((void **)&A_gpu, sizeof(fp32_t) * NI * NK);
    cudaMalloc((void **)&B_gpu, sizeof(fp32_t) * NK * NJ);
    cudaMalloc((void **)&C_gpu, sizeof(fp32_t) * NI * NJ);
    
    cudaMemcpy(A_gpu, A, sizeof(fp32_t) * NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(fp32_t) * NK * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(fp32_t) * NI * NJ, cudaMemcpyHostToDevice);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((NJ + TILE_SIZE - 1) / TILE_SIZE,
              (NI + TILE_SIZE - 1) / TILE_SIZE);

    /* Start timer. */
    polybench_start_instruments;

    // Launch FP32-optimized kernel
    gemm_kernel_fp32<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel execution error: %s\n", cudaGetErrorString(err));
    }

    /* Stop and print timer. */
    printf("GPU Time in seconds (FP32):\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(fp32_t) * NI * NJ, cudaMemcpyDeviceToHost);    
    
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
}

int main(int argc, char *argv[])
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    /* Variable declaration/allocation. */
    fp32_t alpha;
    fp32_t beta;
    POLYBENCH_2D_ARRAY_DECL(A,fp32_t,NI,NK,ni,nk);
    POLYBENCH_2D_ARRAY_DECL(B,fp32_t,NK,NJ,nk,nj);
    POLYBENCH_2D_ARRAY_DECL(C,fp32_t,NI,NJ,ni,nj);
    POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu,fp32_t,NI,NJ,ni,nj);

    init(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
    
    // Copy C to C_outputFromGpu for GPU computation
    memcpy(C_outputFromGpu, C, sizeof(fp32_t) * NI * NJ);
    
    GPU_argv_init();
    
    gemmCuda_fp32(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), 
                  POLYBENCH_ARRAY(C_outputFromGpu), POLYBENCH_ARRAY(C_outputFromGpu));

    #ifdef RUN_ON_CPU
        /* Start timer. */
        polybench_start_instruments;

        gemm(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
        
        /* Stop and print timer. */
        printf("CPU Time in seconds:\n");
        polybench_stop_instruments;
        polybench_print_instruments;
    
        compareResults(ni, nj, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));
    #endif

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);  
    POLYBENCH_FREE_ARRAY(C);  
    POLYBENCH_FREE_ARRAY(C_outputFromGpu); 

    return 0;
}

#include "common/polybench.c"