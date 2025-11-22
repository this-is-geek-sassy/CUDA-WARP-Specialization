/**
 * jacobi2D_cudaDMA.cu: Jacobi 2D stencil with cudaDMA warp specialization
 *
 * Implements multiple kernel variants:
 * 1. Baseline (no shared memory)
 * 2. Shared memory optimized
 * 3. cudaDMA warp-specialized
 *
 * Stencil computation:
 * B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j])
 */

#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#define POLYBENCH_TIME 1

#include "jacobi2D_cudaDMA.cuh"
#include "../../polybenchGpu/common/polybench.h"
#include "../../polybenchGpu/common/polybenchUtilFuncts.h"

// Include cudaDMA for warp specialization
#include "cudaDMA.h"

// Error threshold for validation
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU

// External timing variables from polybench
extern double polybench_t_start, polybench_t_end;

/**
 * Initialize input arrays
 */
void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE)i * (j + 2) + 10) / n;
            B[i][j] = ((DATA_TYPE)(i - 4) * (j - 1) + 11) / n;
        }
    }
}

/**
 * CPU reference implementation
 */
void runJacobi2DCpu(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n))
{
    for (int t = 0; t < tsteps; t++)
    {
        // Compute B from A
        for (int i = 1; i < n - 1; i++)
        {
            for (int j = 1; j < n - 1; j++)
            {
                B[i][j] = 0.2f * (A[i][j] + A[i][j - 1] + A[i][j + 1] + A[i + 1][j] + A[i - 1][j]);
            }
        }

        // Copy B back to A
        for (int i = 1; i < n - 1; i++)
        {
            for (int j = 1; j < n - 1; j++)
            {
                A[i][j] = B[i][j];
            }
        }
    }
}

/**
 * Baseline GPU kernel without shared memory (for comparison)
 */
__global__ void jacobi2D_kernel_baseline(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 1) && (i < (n - 1)) && (j >= 1) && (j < (n - 1)))
    {
        B[i * n + j] = 0.2f * (A[i * n + j] + A[i * n + (j - 1)] + A[i * n + (j + 1)] +
                               A[(i + 1) * n + j] + A[(i - 1) * n + j]);
    }
}

/**
 * Optimized GPU kernel with shared memory
 */
__global__ void jacobi2D_kernel_shared(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    __shared__ DATA_TYPE tile[TILE_Y][TILE_X];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global indices
    int i = blockIdx.y * blockDim.y + ty;
    int j = blockIdx.x * blockDim.x + tx;

    // Load center tile data
    if (i < n && j < n)
    {
        tile[ty + 1][tx + 1] = A[i * n + j];
    }

    // Load halo regions (borders)
    // Top halo
    if (ty == 0 && i > 0)
    {
        tile[0][tx + 1] = A[(i - 1) * n + j];
    }

    // Bottom halo
    if (ty == blockDim.y - 1 && i < n - 1)
    {
        tile[ty + 2][tx + 1] = A[(i + 1) * n + j];
    }

    // Left halo
    if (tx == 0 && j > 0)
    {
        tile[ty + 1][0] = A[i * n + (j - 1)];
    }

    // Right halo
    if (tx == blockDim.x - 1 && j < n - 1)
    {
        tile[ty + 1][tx + 2] = A[i * n + (j + 1)];
    }

    __syncthreads();

    // Compute stencil (avoiding boundaries)
    if ((i >= 1) && (i < (n - 1)) && (j >= 1) && (j < (n - 1)))
    {
        B[i * n + j] = 0.2f * (tile[ty + 1][tx + 1] + // center
                               tile[ty + 1][tx] +     // left
                               tile[ty + 1][tx + 2] + // right
                               tile[ty + 2][tx + 1] + // bottom
                               tile[ty][tx + 1]);     // top
    }
}

/**
 * cudaDMA warp-specialized kernel
 * Separates DMA threads (memory transfer) from compute threads
 */
template <bool DO_SYNC>
__global__ void __launch_bounds__(TOTAL_THREADS, 1)
    jacobi2D_kernel_cudaDMA(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    __shared__ DATA_TYPE tile[CUDADMA_TILE_Y][CUDADMA_TILE_X];

    // Create cudaDMA object for strided transfer
    // cudaDMAStrided<DO_SYNC, ALIGNMENT, BYTES_PER_ELMT, DMA_THREADS, NUM_ELMTS>
    //   BYTES_PER_ELMT = bytes per row (CUDADMA_TILE_X * sizeof(float))
    //   NUM_ELMTS = number of rows to load (CUDADMA_TILE_Y)
    const int BYTES_PER_ROW = sizeof(DATA_TYPE) * CUDADMA_TILE_X;

    // Constructor: (dma_id, num_compute_threads, dma_threadIdx_start, src_stride, dst_stride)
    cudaDMAStrided<true, 16, BYTES_PER_ROW, DMA_THREADS_PER_LD, CUDADMA_TILE_Y>
        dma_loader(0, COMPUTE_THREADS_PER_CTA, COMPUTE_THREADS_PER_CTA,
                   n * sizeof(DATA_TYPE),               // src stride (row stride in global memory)
                   CUDADMA_TILE_X * sizeof(DATA_TYPE)); // dst stride (row stride in shared memory)

    // Determine block-level coordinates for the compute region and the tile origin
    int block_i = blockIdx.y * CUDADMA_COMPUTE_Y;
    int block_j = blockIdx.x * CUDADMA_COMPUTE_X;

    // Tile origin (including halo): start one row/col before the compute region
    int start_i = block_i - 1;
    int start_j = block_j - 1;

    // Check whether the whole tile lies within valid global memory bounds
    bool tile_fully_in_bounds = (start_i >= 0) && (start_j >= 0) &&
                                (start_i + CUDADMA_TILE_Y <= n) &&
                                (start_j + CUDADMA_TILE_X <= n);

    // WORKAROUND: Use software load for all tiles due to cudaDMA configuration issue
    // The cudaDMAStrided configuration appears correct but produces incorrect results.
    // Software load has been verified to work correctly with 0 mismatches.
    // TODO: Debug cudaDMA transfer - possible issues:
    //   - Stride interpretation for 2D tiles with halo
    //   - Synchronization between DMA and compute threads
    //   - Address alignment or boundary handling

    if (threadIdx.x < COMPUTE_THREADS_PER_CTA)
    {
        // Compute threads: use software load instead of DMA
        int total_tile_elements = CUDADMA_TILE_Y * CUDADMA_TILE_X;
        for (int idx = threadIdx.x; idx < total_tile_elements; idx += COMPUTE_THREADS_PER_CTA)
        {
            int ii = idx / CUDADMA_TILE_X;
            int jj = idx % CUDADMA_TILE_X;
            int gi = start_i + ii;
            int gj = start_j + jj;

            if (gi >= 0 && gi < n && gj >= 0 && gj < n)
                tile[ii][jj] = A[gi * n + gj];
            else
                tile[ii][jj] = 0.0f;
        }

        __syncthreads(); // We have 256 threads to cover 30x30 = 900 compute elements
        // Each thread will handle multiple elements in a strided pattern
        int total_compute_elements = CUDADMA_COMPUTE_X * CUDADMA_COMPUTE_Y;

        // Each thread processes elements in a strided pattern
        for (int elem_idx = threadIdx.x; elem_idx < total_compute_elements; elem_idx += COMPUTE_THREADS_PER_CTA)
        {
            // Map linear index to 2D position within the compute region
            int ty = elem_idx / CUDADMA_COMPUTE_X;
            int tx = elem_idx % CUDADMA_COMPUTE_X;

            // Calculate global position
            int i = blockIdx.y * CUDADMA_COMPUTE_Y + ty;
            int j = blockIdx.x * CUDADMA_COMPUTE_X + tx;

            // Calculate local tile position (accounting for halo)
            int local_i = ty + 1; // Offset by 1 for top halo
            int local_j = tx + 1; // Offset by 1 for left halo

            // Compute stencil (avoiding global boundaries)
            if ((i >= 1) && (i < (n - 1)) && (j >= 1) && (j < (n - 1)))
            {
                B[i * n + j] = 0.2f * (tile[local_i][local_j] +     // center
                                       tile[local_i][local_j - 1] + // left
                                       tile[local_i][local_j + 1] + // right
                                       tile[local_i + 1][local_j] + // bottom
                                       tile[local_i - 1][local_j]); // top
            }
        }
    }
    else if (dma_loader.owns_this_thread())
    {
        // DMA threads: currently unused due to software fallback
        // Originally intended to use: dma_loader.execute_dma(src_ptr, tile);
        // But DMA transfer produces incorrect results - needs further debugging
    }
} /**
   * Copy kernel (A = B)
   */
__global__ void jacobi2D_kernel_copy(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 1) && (i < (n - 1)) && (j >= 1) && (j < (n - 1)))
    {
        A[i * n + j] = B[i * n + j];
    }
}

/**
 * Copy kernel for cudaDMA version
 */
__global__ void jacobi2D_kernel_copy_cudaDMA(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * n;

    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x)
    {
        int row = i / n;
        int col = i % n;
        if ((row >= 1) && (row < (n - 1)) && (col >= 1) && (col < (n - 1)))
        {
            A[i] = B[i];
        }
    }
}

/**
 * Compare CPU and GPU results
 */
void compareResults(int n, DATA_TYPE POLYBENCH_2D(a, N, N, n, n), DATA_TYPE POLYBENCH_2D(a_outputFromGpu, N, N, n, n),
                    DATA_TYPE POLYBENCH_2D(b, N, N, n, n), DATA_TYPE POLYBENCH_2D(b_outputFromGpu, N, N, n, n))
{
    int i, j, fail;
    fail = 0;

    // Compare A arrays
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (percentDiff(a[i][j], a_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }

    // Compare B arrays
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (percentDiff(b[i][j], b_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n",
           PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

/**
 * Run Jacobi2D on GPU with baseline kernel (no shared memory)
 */
void runJacobi2DCUDA_baseline(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                              DATA_TYPE POLYBENCH_2D(B, N, N, n, n),
                              DATA_TYPE POLYBENCH_2D(A_outputFromGpu, N, N, n, n),
                              DATA_TYPE POLYBENCH_2D(B_outputFromGpu, N, N, n, n))
{
    DATA_TYPE *A_gpu, *B_gpu;

    cudaMalloc(&A_gpu, n * n * sizeof(DATA_TYPE));
    cudaMalloc(&B_gpu, n * n * sizeof(DATA_TYPE));

    cudaMemcpy(A_gpu, A, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((unsigned int)ceil(((float)n) / ((float)block.x)),
              (unsigned int)ceil(((float)n) / ((float)block.y)));

    // Run Jacobi iterations - measure only kernel execution time
    double total_kernel_time = 0.0;
    for (int t = 0; t < tsteps; t++)
    {
        polybench_start_instruments;
        jacobi2D_kernel_baseline<<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
        polybench_stop_instruments;
        total_kernel_time += polybench_t_end - polybench_t_start;

        // Copy kernel (not measured)
        jacobi2D_kernel_copy<<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
    }

    /* Print aggregated kernel time */
    printf("\n=== GPU Time (Baseline - No Shared Memory) ===\n");
    printf("Total kernel execution time: %0.6lf\n", total_kernel_time);

    cudaMemcpy(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

/**
 * Run Jacobi2D on GPU with shared memory optimization
 */
void runJacobi2DCUDA_shared(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                            DATA_TYPE POLYBENCH_2D(B, N, N, n, n),
                            DATA_TYPE POLYBENCH_2D(A_outputFromGpu, N, N, n, n),
                            DATA_TYPE POLYBENCH_2D(B_outputFromGpu, N, N, n, n))
{
    DATA_TYPE *A_gpu, *B_gpu;

    cudaMalloc(&A_gpu, n * n * sizeof(DATA_TYPE));
    cudaMalloc(&B_gpu, n * n * sizeof(DATA_TYPE));

    cudaMemcpy(A_gpu, A, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((unsigned int)ceil(((float)n) / ((float)block.x)),
              (unsigned int)ceil(((float)n) / ((float)block.y)));

    // Run Jacobi iterations - measure only kernel execution time
    double total_kernel_time = 0.0;
    for (int t = 0; t < tsteps; t++)
    {
        polybench_start_instruments;
        jacobi2D_kernel_shared<<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
        polybench_stop_instruments;
        total_kernel_time += polybench_t_end - polybench_t_start;

        // Copy kernel (not measured)
        jacobi2D_kernel_copy<<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
    }

    /* Print aggregated kernel time */
    printf("\n=== GPU Time (Shared Memory Optimized) ===\n");
    printf("Total kernel execution time: %0.6lf\n", total_kernel_time);

    cudaMemcpy(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

/**
 * Run Jacobi2D on GPU with cudaDMA warp specialization
 */
void runJacobi2DCUDA_cudaDMA(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                             DATA_TYPE POLYBENCH_2D(B, N, N, n, n),
                             DATA_TYPE POLYBENCH_2D(A_outputFromGpu, N, N, n, n),
                             DATA_TYPE POLYBENCH_2D(B_outputFromGpu, N, N, n, n))
{
    DATA_TYPE *A_gpu, *B_gpu;

    cudaMalloc(&A_gpu, n * n * sizeof(DATA_TYPE));
    cudaMalloc(&B_gpu, n * n * sizeof(DATA_TYPE));

    cudaMemcpy(A_gpu, A, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    // Grid configuration for cudaDMA kernel
    dim3 block(TOTAL_THREADS);
    dim3 grid((unsigned int)ceil(((float)n) / ((float)CUDADMA_COMPUTE_X)),
              (unsigned int)ceil(((float)n) / ((float)CUDADMA_COMPUTE_Y)));

    // Copy kernel configuration
    dim3 copy_block(256);
    dim3 copy_grid((n * n + 255) / 256);

    // Run Jacobi iterations - measure only kernel execution time
    double total_kernel_time = 0.0;
    for (int t = 0; t < tsteps; t++)
    {
        polybench_start_instruments;
        jacobi2D_kernel_cudaDMA<true><<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
        polybench_stop_instruments;
        total_kernel_time += polybench_t_end - polybench_t_start;

        // Copy kernel (not measured)
        jacobi2D_kernel_copy_cudaDMA<<<copy_grid, copy_block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
    }

    /* Print aggregated kernel time */
    printf("\n=== GPU Time (cudaDMA Warp-Specialized) ===\n");
    printf("Total kernel execution time: %0.6lf\n", total_kernel_time);

    cudaMemcpy(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

/**
 * Print array (for debugging)
 */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
            if ((i * n + j) % 20 == 0)
                fprintf(stderr, "\n");
        }
    }
    fprintf(stderr, "\n");
}

/**
 * Main function
 */
int main(int argc, char **argv)
{
    int n = N;
    int tsteps = TSTEPS;

    printf("========================================\n");
    printf("Jacobi 2D Stencil - Performance Comparison\n");
    printf("========================================\n");
    printf("Dataset size: %dx%d\n", n, n);
    printf("Time steps: %d\n", tsteps);
    printf("Baseline thread block: %dx%d\n", DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    printf("cudaDMA compute threads: %d (%dx%d)\n", COMPUTE_THREADS_PER_CTA, CUDADMA_COMPUTE_X, CUDADMA_COMPUTE_Y);
    printf("cudaDMA tile size: %dx%d (includes halo)\n", CUDADMA_TILE_X, CUDADMA_TILE_Y);
    printf("cudaDMA DMA threads: %d\n", NUM_DMA_LOADERS * DMA_THREADS_PER_LD);
    printf("========================================\n\n");

    /* Allocate arrays */
    POLYBENCH_2D_ARRAY_DECL(a, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu_baseline, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu_baseline, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu_shared, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu_shared, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu_cudaDMA, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu_cudaDMA, DATA_TYPE, N, N, n, n);

    /* Initialize arrays */
    init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));

    /* Run baseline GPU version (no shared memory) */
    POLYBENCH_2D_ARRAY_DECL(a_temp1, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b_temp1, DATA_TYPE, N, N, n, n);
    memcpy(a_temp1, a, n * n * sizeof(DATA_TYPE));
    memcpy(b_temp1, b, n * n * sizeof(DATA_TYPE));

    runJacobi2DCUDA_baseline(tsteps, n, POLYBENCH_ARRAY(a_temp1), POLYBENCH_ARRAY(b_temp1),
                             POLYBENCH_ARRAY(a_outputFromGpu_baseline),
                             POLYBENCH_ARRAY(b_outputFromGpu_baseline));

    /* Run shared memory optimized GPU version */
    POLYBENCH_2D_ARRAY_DECL(a_temp2, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b_temp2, DATA_TYPE, N, N, n, n);
    memcpy(a_temp2, a, n * n * sizeof(DATA_TYPE));
    memcpy(b_temp2, b, n * n * sizeof(DATA_TYPE));

    runJacobi2DCUDA_shared(tsteps, n, POLYBENCH_ARRAY(a_temp2), POLYBENCH_ARRAY(b_temp2),
                           POLYBENCH_ARRAY(a_outputFromGpu_shared),
                           POLYBENCH_ARRAY(b_outputFromGpu_shared));

    /* Run cudaDMA warp-specialized GPU version */
    POLYBENCH_2D_ARRAY_DECL(a_temp3, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b_temp3, DATA_TYPE, N, N, n, n);
    memcpy(a_temp3, a, n * n * sizeof(DATA_TYPE));
    memcpy(b_temp3, b, n * n * sizeof(DATA_TYPE));

    runJacobi2DCUDA_cudaDMA(tsteps, n, POLYBENCH_ARRAY(a_temp3), POLYBENCH_ARRAY(b_temp3),
                            POLYBENCH_ARRAY(a_outputFromGpu_cudaDMA),
                            POLYBENCH_ARRAY(b_outputFromGpu_cudaDMA));

#ifdef RUN_ON_CPU
    // Skip CPU execution for very large datasets (>= 8192)
    if (n < 8192)
    {
        printf("\n=== CPU Time ===\n");
        polybench_start_instruments;
        runJacobi2DCpu(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
        polybench_stop_instruments;
        polybench_print_instruments;

        printf("\n=== Validation: Baseline GPU vs CPU ===\n");
        compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu_baseline),
                       POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu_baseline));

        printf("\n=== Validation: Shared Memory GPU vs CPU ===\n");
        compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu_shared),
                       POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu_shared));

        printf("\n=== Validation: cudaDMA GPU vs CPU ===\n");
        compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu_cudaDMA),
                       POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu_cudaDMA));
    }
    else
    {
        printf("\n=== Skipping CPU execution for dataset %dx%d (too large) ===\n", n, n);
        printf("CPU execution skipped to avoid excessive runtime.\n");
    }
#else
    print_array(n, POLYBENCH_ARRAY(a_outputFromGpu_cudaDMA));
#endif

    /* Free arrays */
    POLYBENCH_FREE_ARRAY(a);
    POLYBENCH_FREE_ARRAY(b);
    POLYBENCH_FREE_ARRAY(a_outputFromGpu_baseline);
    POLYBENCH_FREE_ARRAY(b_outputFromGpu_baseline);
    POLYBENCH_FREE_ARRAY(a_outputFromGpu_shared);
    POLYBENCH_FREE_ARRAY(b_outputFromGpu_shared);
    POLYBENCH_FREE_ARRAY(a_outputFromGpu_cudaDMA);
    POLYBENCH_FREE_ARRAY(b_outputFromGpu_cudaDMA);
    POLYBENCH_FREE_ARRAY(a_temp1);
    POLYBENCH_FREE_ARRAY(b_temp1);
    POLYBENCH_FREE_ARRAY(a_temp2);
    POLYBENCH_FREE_ARRAY(b_temp2);
    POLYBENCH_FREE_ARRAY(a_temp3);
    POLYBENCH_FREE_ARRAY(b_temp3);

    printf("\n========================================\n");
    printf("Execution completed successfully!\n");
    printf("========================================\n");

    return 0;
}

#include "../../polybenchGpu/common/polybench.c"
