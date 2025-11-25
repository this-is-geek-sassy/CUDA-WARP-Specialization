#ifndef CUDA_DMA_SAS_H
#define CUDA_DMA_SAS_H

#include<cuda.h>
#include "../headers/cudaDMA.h"

// cudaDMA-based GEMM kernel with warp specialization  --- double buffering
template<
  unsigned int TILE_SIZE,
  unsigned int COMPUTE_THREADS_PER_CTA,
  unsigned int DMA_THREADS_PER_LD,
  unsigned int NUM_DMA_LOADERS,
  unsigned int TOTAL_THREADS
>
__global__ void dgemm_cuda_dma_sas(int ni, int nj, int nk, float alpha, float beta, 
                                         float *a, float *b, float *c)
{
    // Double buffering: two buffers for A and two for B
    __shared__ float As_0[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs_0[TILE_SIZE][TILE_SIZE];
    __shared__ float As_1[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs_1[TILE_SIZE][TILE_SIZE];
    
    cudaDMAStrided<true, 16, 128, DMA_THREADS_PER_LD, TILE_SIZE>
        dma_ld_a(0, COMPUTE_THREADS_PER_CTA, COMPUTE_THREADS_PER_CTA,
                 nk * sizeof(float), TILE_SIZE * sizeof(float));
    
    cudaDMAStrided<true, 16, 128, DMA_THREADS_PER_LD, TILE_SIZE>
        dma_ld_b(1, COMPUTE_THREADS_PER_CTA, 
                 COMPUTE_THREADS_PER_CTA + DMA_THREADS_PER_LD,
                 nj * sizeof(float), TILE_SIZE * sizeof(float));
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int numTiles = (nk + TILE_SIZE - 1) / TILE_SIZE;
    
    // Compute threads
    if (threadIdx.x < COMPUTE_THREADS_PER_CTA)
    {
        int thread_id = threadIdx.x;
        int elements_per_thread = (TILE_SIZE * TILE_SIZE) / COMPUTE_THREADS_PER_CTA; // 4
        
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        // Load first tile into buffer 0
        dma_ld_a.start_async_dma();
        dma_ld_b.start_async_dma();
        
        for (int t = 0; t < numTiles; t++) 
        {
            // Determine which buffer to use for current computation (ping-pong)
            int curr_buf = t & 1;  // 0 or 1
            // int next_buf = 1 - curr_buf;  // Next buffer for DMA load (for readability)
            
            // Wait for current tile to be loaded
            dma_ld_a.wait_for_dma_finish();
            dma_ld_b.wait_for_dma_finish();
            
            // Signal next tile load into alternate buffer (if not last iteration)
            if (t < numTiles - 1) {
                dma_ld_a.start_async_dma();
                dma_ld_b.start_async_dma();
            }
            
            // Compute on current tile while next tile is being loaded
            for (int elem = 0; elem < elements_per_thread; elem++)
            {
                int linear_idx = thread_id * elements_per_thread + elem;
                int ty = linear_idx / TILE_SIZE;
                int tx = linear_idx % TILE_SIZE;
                
                // Select current buffer based on curr_buf
                if (curr_buf == 0) {
                    #pragma unroll
                    for (int k = 0; k < TILE_SIZE; k++) {
                        sums[elem] += As_0[ty][k] * Bs_0[k][tx];
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < TILE_SIZE; k++) {
                        sums[elem] += As_1[ty][k] * Bs_1[k][tx];
                    }
                }
            }
        }
        
        #pragma unroll
        // Write results
        for (int elem = 0; elem < elements_per_thread; elem++)
        {
            int linear_idx = thread_id * elements_per_thread + elem;
            int ty = linear_idx / TILE_SIZE;
            int tx = linear_idx % TILE_SIZE;
            
            int row = by * TILE_SIZE + ty;
            int col = bx * TILE_SIZE + tx;
            
            if (row < ni && col < nj) {
                c[row * nj + col] = beta * c[row * nj + col] + alpha * sums[elem];
            }
        }
    }
    // DMA threads for A
    else if (dma_ld_a.owns_this_thread())
    {
        for (int t = 0; t < numTiles; t++)
        {
            int aRow = by * TILE_SIZE;
            int aCol = t * TILE_SIZE;
            
            // Determine which buffer to load into (ping-pong)
            int buf_idx = t & 1;
            
            // Boundary check for source pointer
            if (aRow < ni && aCol < nk) {
                float *src_ptr = &a[aRow * nk + aCol];
                // Load into alternating buffer
                if (buf_idx == 0) {
                    dma_ld_a.execute_dma(src_ptr, As_0);
                } else {
                    dma_ld_a.execute_dma(src_ptr, As_1);
                }
            } else {
                // Still need to participate in synchronization even if out of bounds
                dma_ld_a.wait_for_dma_start();
                dma_ld_a.finish_async_dma();
            }
        }
    }
    // DMA threads for B
    else if (dma_ld_b.owns_this_thread())
    {
        for (int t = 0; t < numTiles; t++)
        {
            int bRow = t * TILE_SIZE;
            int bCol = bx * TILE_SIZE;
            
            // Determine which buffer to load into (ping-pong)
            int buf_idx = t & 1;
            
            // Boundary check for source pointer
            if (bRow < nk && bCol < nj) {
                float *src_ptr = &b[bRow * nj + bCol];
                // Load into alternating buffer
                if (buf_idx == 0) {
                    dma_ld_b.execute_dma(src_ptr, Bs_0);
                } else {
                    dma_ld_b.execute_dma(src_ptr, Bs_1);
                }
            } else {
                // Still need to participate in synchronization
                dma_ld_b.wait_for_dma_start();
                dma_ld_b.finish_async_dma();
            }
        }
    }
}

#endif // CUDA_DMA_SAS_H
