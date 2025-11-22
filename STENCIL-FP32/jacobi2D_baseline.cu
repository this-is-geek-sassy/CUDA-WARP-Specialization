/**
 * jacobi2D_baseline.cu: Baseline Jacobi 2D stencil with shared memory optimization
 *
 * Implements 5-point stencil computation:
 * B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j])
 *
 * Optimizations:
 * - Shared memory tiling to reduce global memory accesses
 * - Halo region loading for stencil computation
 * - Coalesced memory access patterns
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

#include "jacobi2D_baseline.cuh"
#include "../../polybenchGpu/common/polybench.h"
#include "../../polybenchGpu/common/polybenchUtilFuncts.h"

// Error threshold for validation
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU

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
 * Each block loads a tile of data including halo regions into shared memory
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
 * Texture memory optimized GPU kernel
 * Uses texture memory for spatial locality and caching benefits
 */
__global__ void jacobi2D_kernel_texture(int n, cudaTextureObject_t texA, DATA_TYPE *B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 1) && (i < (n - 1)) && (j >= 1) && (j < (n - 1)))
    {
        // Read from texture memory (provides hardware caching and 2D spatial locality)
        DATA_TYPE center = tex2D<DATA_TYPE>(texA, j, i);
        DATA_TYPE left = tex2D<DATA_TYPE>(texA, j - 1, i);
        DATA_TYPE right = tex2D<DATA_TYPE>(texA, j + 1, i);
        DATA_TYPE top = tex2D<DATA_TYPE>(texA, j, i - 1);
        DATA_TYPE bottom = tex2D<DATA_TYPE>(texA, j, i + 1);

        B[i * n + j] = 0.2f * (center + left + right + top + bottom);
    }
}

/**
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

    // Allocate device memory
    cudaMalloc(&A_gpu, n * n * sizeof(DATA_TYPE));
    cudaMalloc(&B_gpu, n * n * sizeof(DATA_TYPE));

    // Copy data to device
    cudaMemcpy(A_gpu, A, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    // Setup execution configuration
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((unsigned int)ceil(((float)n) / ((float)block.x)),
              (unsigned int)ceil(((float)n) / ((float)block.y)));

    /* Start timer */
    polybench_start_instruments;

    // Run Jacobi iterations
    for (int t = 0; t < tsteps; t++)
    {
        jacobi2D_kernel_baseline<<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
        jacobi2D_kernel_copy<<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
    }

    /* Stop and print timer */
    printf("\n=== GPU Time (Baseline - No Shared Memory) ===\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    // Copy results back
    cudaMemcpy(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

/**
 * Run Jacobi2D on GPU with texture memory optimization
 */
void runJacobi2DCUDA_texture(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                             DATA_TYPE POLYBENCH_2D(B, N, N, n, n),
                             DATA_TYPE POLYBENCH_2D(A_outputFromGpu, N, N, n, n),
                             DATA_TYPE POLYBENCH_2D(B_outputFromGpu, N, N, n, n))
{
    DATA_TYPE *A_gpu, *B_gpu;
    cudaArray *cuArray;
    cudaTextureObject_t texA = 0;

    // Allocate device memory
    cudaMalloc(&A_gpu, n * n * sizeof(DATA_TYPE));
    cudaMalloc(&B_gpu, n * n * sizeof(DATA_TYPE));

    // Create channel descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<DATA_TYPE>();

    // Allocate CUDA array for texture
    cudaMallocArray(&cuArray, &channelDesc, n, n);

    // Copy data to CUDA array
    cudaMemcpy2DToArray(cuArray, 0, 0, A, n * sizeof(DATA_TYPE),
                        n * sizeof(DATA_TYPE), n, cudaMemcpyHostToDevice);

    // Also copy A to device memory (needed for the copy kernel)
    cudaMemcpy(A_gpu, A, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    // Specify texture resource
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaCreateTextureObject(&texA, &resDesc, &texDesc, NULL);

    // Copy B array to device
    cudaMemcpy(B_gpu, B, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    // Setup execution configuration
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((unsigned int)ceil(((float)n) / ((float)block.x)),
              (unsigned int)ceil(((float)n) / ((float)block.y)));

    /* Start timer */
    polybench_start_instruments;

    // Run Jacobi iterations
    for (int t = 0; t < tsteps; t++)
    {
        // Compute B from texture (which contains A)
        jacobi2D_kernel_texture<<<grid, block>>>(n, texA, B_gpu);
        cudaDeviceSynchronize();

        // Only copy interior values (matching the copy kernel behavior)
        jacobi2D_kernel_copy<<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();

        // Update texture with new A values for next iteration
        cudaMemcpy2DToArray(cuArray, 0, 0, A_gpu, n * sizeof(DATA_TYPE),
                            n * sizeof(DATA_TYPE), n, cudaMemcpyDeviceToDevice);
    }

    /* Stop and print timer */
    printf("\n=== GPU Time (Texture Memory Optimized) ===\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    // Copy results back (A contains the final values after copy-back)
    cudaMemcpy(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaDestroyTextureObject(texA);
    cudaFreeArray(cuArray);
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

    // Allocate device memory
    cudaMalloc(&A_gpu, n * n * sizeof(DATA_TYPE));
    cudaMalloc(&B_gpu, n * n * sizeof(DATA_TYPE));

    // Copy data to device
    cudaMemcpy(A_gpu, A, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    // Setup execution configuration
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((unsigned int)ceil(((float)n) / ((float)block.x)),
              (unsigned int)ceil(((float)n) / ((float)block.y)));

    /* Start timer */
    polybench_start_instruments;

    // Run Jacobi iterations
    for (int t = 0; t < tsteps; t++)
    {
        jacobi2D_kernel_shared<<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
        jacobi2D_kernel_copy<<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
    }

    /* Stop and print timer */
    printf("\n=== GPU Time (Shared Memory Optimized) ===\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    // Copy results back
    cudaMemcpy(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

/**
 * Print array (for debugging)
 */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), FILE *fp = stderr)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            fprintf(fp, DATA_PRINTF_MODIFIER, A[i][j]);
            if ((i * n + j) % 20 == 0)
                fprintf(fp, "\n");
        }
    }
    fprintf(fp, "\n");
}

/**
 * Main function
 */
int main(int argc, char **argv)
{
    /* Retrieve problem size */
    int n = N;
    int tsteps = TSTEPS;

    printf("========================================\n");
    printf("Jacobi 2D Stencil - Baseline with Shared Memory Optimization\n");
    printf("========================================\n");
    printf("Dataset size: %dx%d\n", n, n);
    printf("Time steps: %d\n", tsteps);
    printf("Thread block: %dx%d\n", DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    printf("========================================\n\n");

    /* Allocate arrays */
    POLYBENCH_2D_ARRAY_DECL(a, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu_baseline, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu_baseline, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu_shared, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu_shared, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu_texture, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu_texture, DATA_TYPE, N, N, n, n);

    /* Initialize arrays */
    init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));

    // print a and b array into a file here after init
    FILE *fp_a = fopen("array_a.txt", "w");
    FILE *fp_b = fopen("array_b.txt", "w");

    print_array(n, POLYBENCH_ARRAY(a), fp_a);
    print_array(n, POLYBENCH_ARRAY(b), fp_b);
    fclose(fp_a);
    fclose(fp_b);
    // print_array(n, POLYBENCH_ARRAY(a));
    // print_array(n, POLYBENCH_ARRAY(b));

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

    /* Run texture memory optimized GPU version */
    POLYBENCH_2D_ARRAY_DECL(a_temp3, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b_temp3, DATA_TYPE, N, N, n, n);
    memcpy(a_temp3, a, n * n * sizeof(DATA_TYPE));
    memcpy(b_temp3, b, n * n * sizeof(DATA_TYPE));

    runJacobi2DCUDA_texture(tsteps, n, POLYBENCH_ARRAY(a_temp3), POLYBENCH_ARRAY(b_temp3),
                            POLYBENCH_ARRAY(a_outputFromGpu_texture),
                            POLYBENCH_ARRAY(b_outputFromGpu_texture));

#ifdef RUN_ON_CPU
    // Skip CPU execution for very large datasets (>= 8192)
    if (n < 8192)
    {
        /* Run CPU version */
        printf("\n=== CPU Time ===\n");
        polybench_start_instruments;
        runJacobi2DCpu(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
        polybench_stop_instruments;
        polybench_print_instruments;

        /* Compare results - baseline vs CPU */
        printf("\n=== Validation: Baseline GPU vs CPU ===\n");
        compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu_baseline),
                       POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu_baseline));

        /* Compare results - shared memory vs CPU */
        printf("\n=== Validation: Shared Memory GPU vs CPU ===\n");
        compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu_shared),
                       POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu_shared));

        /* Compare results - texture memory vs CPU */
        printf("\n=== Validation: Texture Memory GPU vs CPU ===\n");
        compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu_texture),
                       POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu_texture));
    }
    else
    {
        printf("\n=== Skipping CPU execution for dataset %dx%d (too large) ===\n", n, n);
        printf("CPU execution skipped to avoid excessive runtime.\n");
    }
#else
    /* Print output to stderr (no dead code elimination) */
    print_array(n, POLYBENCH_ARRAY(a_outputFromGpu_shared));
#endif

    /* Free arrays */
    POLYBENCH_FREE_ARRAY(a);
    POLYBENCH_FREE_ARRAY(b);
    POLYBENCH_FREE_ARRAY(a_outputFromGpu_baseline);
    POLYBENCH_FREE_ARRAY(b_outputFromGpu_baseline);
    POLYBENCH_FREE_ARRAY(a_outputFromGpu_shared);
    POLYBENCH_FREE_ARRAY(b_outputFromGpu_shared);
    POLYBENCH_FREE_ARRAY(a_outputFromGpu_texture);
    POLYBENCH_FREE_ARRAY(b_outputFromGpu_texture);
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
