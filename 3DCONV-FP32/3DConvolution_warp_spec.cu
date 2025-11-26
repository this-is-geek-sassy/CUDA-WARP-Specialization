/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "3DConvolution.cuh"
#include "../gpu_utils.h"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

#define RUN_ON_CPU


void conv3D(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk))
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = 1; i < _PB_NI - 1; ++i) // 0
	{
		for (j = 1; j < _PB_NJ - 1; ++j) // 1
		{
			for (k = 1; k < _PB_NK -1; ++k) // 2
			{
				B[i][j][k] = c11 * A[(i - 1)][(j - 1)][(k - 1)]  +  c13 * A[(i + 1)][(j - 1)][(k - 1)]
					     +   c21 * A[(i - 1)][(j - 1)][(k - 1)]  +  c23 * A[(i + 1)][(j - 1)][(k - 1)]
					     +   c31 * A[(i - 1)][(j - 1)][(k - 1)]  +  c33 * A[(i + 1)][(j - 1)][(k - 1)]
					     +   c12 * A[(i + 0)][(j - 1)][(k + 0)]  +  c22 * A[(i + 0)][(j + 0)][(k + 0)]   
					     +   c32 * A[(i + 0)][(j + 1)][(k + 0)]  +  c11 * A[(i - 1)][(j - 1)][(k + 1)]  
					     +   c13 * A[(i + 1)][(j - 1)][(k + 1)]  +  c21 * A[(i - 1)][(j + 0)][(k + 1)]  
					     +   c23 * A[(i + 1)][(j + 0)][(k + 1)]  +  c31 * A[(i - 1)][(j + 1)][(k + 1)]  
					     +   c33 * A[(i + 1)][(j + 1)][(k + 1)];
			}
		}
	}
}


void init(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk))
{
	int i, j, k;

	for (i = 0; i < ni; ++i)
    	{
		for (j = 0; j < nj; ++j)
		{
			for (k = 0; k < nk; ++k)
			{
				A[i][j][k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
		}
	}
}


void compareResults(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B_outputFromGpu, NI, NJ, NK, ni, nj, nk))
{
	int i, j, k, fail;
	fail = 0;
	
	// Compare result from cpu and gpu
	for (i = 1; i < ni - 1; ++i) // 0
	{
		for (j = 1; j < nj - 1; ++j) // 1
		{
			for (k = 1; k < nk - 1; ++k) // 2
			{
				if (percentDiff(B[i][j][k], B_outputFromGpu[i][j][k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}	
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


__global__ void convolution3D_kernel(int ni, int nj, int nk, DATA_TYPE* A, DATA_TYPE* B, int i)
{
	// Thread indices
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tid = ty * blockDim.x + tx; // Linear thread ID
	int warpId = tid / 32; // Warp ID (0-31 for 1024 threads)
	
	int k = blockIdx.x * blockDim.x + tx;
	int j = blockIdx.y * blockDim.y + ty;

	// Double-buffered shared memory with padding
	// Using 2 buffers to overlap loading and computing
	__shared__ DATA_TYPE tile[2][3][DIM_THREAD_BLOCK_Y + 2][DIM_THREAD_BLOCK_X + 3];

	// Constant coefficients
	const DATA_TYPE c11 = +2, c21 = +5, c31 = -8;
	const DATA_TYPE c12 = -3, c22 = +6, c32 = -9;
	const DATA_TYPE c13 = +4, c23 = +7, c33 = +10;

	int buffer = 0; // Current buffer index

	// Warp specialization: first 16 warps (512 threads) load data
	// Remaining 16 warps (512 threads) wait and then all compute together
	bool isLoader = (warpId < 16);

	// WARP SPECIALIZATION: Only loader warps load data using cp.async
	if (isLoader) {
		// 512 loader threads cooperate to load all data
		// Total elements per slice: 34 x 35 = 1190 elements
		// Total elements for 3 slices: 3 x 1190 = 3570 elements
		// Each loader thread loads: 3570 / 512 ≈ 7 elements
		
		int loaderTid = tid; // Thread ID (0-511 for loader threads)
		int totalElements = 3 * (DIM_THREAD_BLOCK_Y + 2) * (DIM_THREAD_BLOCK_X + 3);
		
		for (int idx = loaderTid; idx < totalElements; idx += 512) {
			// Decode linear index to (slice, dj, dk)
			int slice = idx / ((DIM_THREAD_BLOCK_Y + 2) * (DIM_THREAD_BLOCK_X + 3));
			int remainder = idx % ((DIM_THREAD_BLOCK_Y + 2) * (DIM_THREAD_BLOCK_X + 3));
			int dj = remainder / (DIM_THREAD_BLOCK_X + 3);
			int dk = remainder % (DIM_THREAD_BLOCK_X + 3);
			
			int global_j = blockIdx.y * blockDim.y + dj - 1;
			int global_k = blockIdx.x * blockDim.x + dk - 1;
			int i_offset = (slice == 0) ? (i - 1) : ((slice == 1) ? i : (i + 1));
			
			// Load data with boundary checks using cp.async
			if (i_offset >= 0 && i_offset < ni && global_j >= 0 && global_j < nj && global_k >= 0 && global_k < nk) {
				uint32_t smem_addr = __cvta_generic_to_shared(&tile[buffer][slice][dj][dk]);
				const DATA_TYPE* src = &A[i_offset*(NK * NJ) + global_j*NK + global_k];
				asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr), "l"(src));
			} else {
				tile[buffer][slice][dj][dk] = 0.0f;
			}
		}
	}

	// Commit and wait for all cp.async operations, then synchronize
	asm volatile("cp.async.commit_group;\n");
	asm volatile("cp.async.wait_group 0;\n");
	__syncthreads();

	// ALL warps compute (both loader and non-loader warps participate in computation)
	if ((i < (_PB_NI-1)) && (j < (_PB_NJ-1)) && (k < (_PB_NK-1)) && (i > 0) && (j > 0) && (k > 0))
	{
		// Local indices in shared memory (add 1 for halo offset)
		int sj = ty + 1;
		int sk = tx + 1;

		DATA_TYPE result = 
			// Matching the exact pattern from the original code
			c11 * tile[buffer][0][sj-1][sk-1] + c13 * tile[buffer][2][sj-1][sk-1] +
			c21 * tile[buffer][0][sj-1][sk-1] + c23 * tile[buffer][2][sj-1][sk-1] +
			c31 * tile[buffer][0][sj-1][sk-1] + c33 * tile[buffer][2][sj-1][sk-1] +
			c12 * tile[buffer][1][sj-1][sk] +
			c22 * tile[buffer][1][sj][sk] +
			c32 * tile[buffer][1][sj+1][sk] +
			c11 * tile[buffer][0][sj-1][sk+1] +
			c13 * tile[buffer][2][sj-1][sk+1] +
			c21 * tile[buffer][0][sj][sk+1] +
			c23 * tile[buffer][2][sj][sk+1] +
			c31 * tile[buffer][0][sj+1][sk+1] +
			c33 * tile[buffer][2][sj+1][sk+1];

		B[i*(NK * NJ) + j*NK + k] = result;
	}
}


void convolution3DCuda(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B_outputFromGpu, NI, NJ, NK, ni, nj, nk))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NK) / ((float)block.x) )), (size_t)(ceil( ((float)NJ) / ((float)block.y) )));
	
	/* Start timer. */
  	polybench_start_instruments;

	int i;
	for (i = 1; i < _PB_NI - 1; ++i) // 0
	{
		convolution3D_kernel<<< grid, block >>>(ni, nj, nk, A_gpu, B_gpu, i);
	}

	cudaDeviceSynchronize();
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
	
	cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyDeviceToHost);
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj, int nk,
		 DATA_TYPE POLYBENCH_3D(B,NI,NJ,NK,ni,nj,nk))
{
  int i, j, k;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) 
	for (k = 0; k < nk; k++)
	{
	fprintf (stderr, DATA_PRINTF_MODIFIER, B[i][j][k]);
	if ((i * (nj*nk) + j*nk + k) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	int ni = NI;
	int nj = NJ;
	int nk = NK;

	POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,NK,ni,nj,nk);
	POLYBENCH_3D_ARRAY_DECL(B,DATA_TYPE,NI,NJ,NK,ni,nj,nk);
	POLYBENCH_3D_ARRAY_DECL(B_outputFromGpu,DATA_TYPE,NI,NJ,NK,ni,nj,nk);

	init(ni, nj, nk, POLYBENCH_ARRAY(A));
	
	GPU_argv_init();

	convolution3DCuda(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		conv3D(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(ni, nj, nk, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(ni, nj, nk, POLYBENCH_ARRAY(B_outputFromGpu));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(B_outputFromGpu);

    	return 0;
}

#include "../../common/polybench.c"

