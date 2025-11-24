/**
 * jacobi2D.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <cooperative_groups.h>

#define POLYBENCH_TIME 1

#include "jacobi2D.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

namespace cg = cooperative_groups;

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size defined in jacobi2D.cuh */

#define RUN_ON_CPU


void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			A[i][j] = ((DATA_TYPE) i*(j+2) + 10) / N;
			B[i][j] = ((DATA_TYPE) (i-4)*(j-1) + 11) / N;
		}
	}
}


void runJacobi2DCpu(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
	for (int t = 0; t < _PB_TSTEPS; t++)
	{
    		for (int i = 1; i < _PB_N - 1; i++)
		{
			for (int j = 1; j < _PB_N - 1; j++)
			{
	  			B[i][j] = 0.2f * (A[i][j] + A[i][(j-1)] + A[i][(1+j)] + A[(1+i)][j] + A[(i-1)][j]);
			}
		}
		
    		for (int i = 1; i < _PB_N-1; i++)
		{
			for (int j = 1; j < _PB_N-1; j++)
			{
	  			A[i][j] = B[i][j];
			}
		}
	}
}


__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	// TRUE Warp Specialization using memory fences and atomic counters
	// Producer warps (0-3): Load data into shared memory
	// Consumer warps (4-15): Compute results from shared memory
	__shared__ DATA_TYPE tile[DIM_THREAD_BLOCK_Y + 2][DIM_THREAD_BLOCK_X + 2];
	__shared__ int load_counter;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = blockIdx.y * blockDim.y + ty;
	int j = blockIdx.x * blockDim.x + tx;
	
	int tid = ty * blockDim.x + tx;
	int warpId = tid / 32;
	int laneId = tid % 32;
	
	const int NUM_PRODUCER_WARPS = 4;  // Warps 0-3 are producers
	const int TILE_WIDTH = DIM_THREAD_BLOCK_X + 2;  // 34
	const int TILE_HEIGHT = DIM_THREAD_BLOCK_Y + 2; // 18
	const int TILE_SIZE = TILE_WIDTH * TILE_HEIGHT; // 612 elements
	
	// Initialize counter
	if (tid == 0) load_counter = 0;
	__syncthreads();
	
	// === PRODUCER WARPS: Load data with strided access ===
	if (warpId < NUM_PRODUCER_WARPS) {
		const int NUM_PRODUCERS = NUM_PRODUCER_WARPS * 32; // 128 threads
		
		// Each producer loads multiple elements using strided pattern
		for (int idx = tid; idx < TILE_SIZE; idx += NUM_PRODUCERS) {
			int tile_y = idx / TILE_WIDTH;
			int tile_x = idx % TILE_WIDTH;
			
			// Map to global coordinates (account for halo)
			int global_y = blockIdx.y * DIM_THREAD_BLOCK_Y + tile_y - 1;
			int global_x = blockIdx.x * DIM_THREAD_BLOCK_X + tile_x - 1;
			
			// Load from global memory with bounds checking
			if (global_y >= 0 && global_y < n && global_x >= 0 && global_x < n) {
				tile[tile_y][tile_x] = A[global_y * N + global_x];
			} else {
				tile[tile_y][tile_x] = 0.0f;
			}
		}
		
		// Ensure all producer writes are visible across block
		__threadfence_block();
		
		// Last thread in producer warps increments counter
		if (laneId == 31) {
			atomicAdd(&load_counter, 1);
		}
	}
	
	// === CONSUMER WARPS: Wait for data, then compute ===
	// Wait for all producer warps to finish
	if (warpId >= NUM_PRODUCER_WARPS) {
		// Busy-wait until all producers signal completion
		while (atomicAdd(&load_counter, 0) < NUM_PRODUCER_WARPS);
	}
	
	// Ensure loads are visible before computation
	__threadfence_block();
	
	// ALL warps now compute (including producers, after they finish loading)
	if (warpId < NUM_PRODUCER_WARPS) {
		// Producers wait for all producers to finish before computing
		while (atomicAdd(&load_counter, 0) < NUM_PRODUCER_WARPS);
		__threadfence_block();
	}
	
	// Compute stencil from shared memory
	if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
	{
		B[i*N + j] = 0.2f * (tile[ty + 1][tx + 1] + 
		                      tile[ty + 1][tx] + 
		                      tile[ty + 1][tx + 2] + 
		                      tile[ty + 2][tx + 1] + 
		                      tile[ty][tx + 1]);
	}
}


__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = blockIdx.y * blockDim.y + ty;
	int j = blockIdx.x * blockDim.x + tx;
	
	// Simple copy - no warp specialization needed for this kernel
	if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
	{
		A[i*N + j] = B[i*N + j];
	}
}


void compareResults(int n, DATA_TYPE POLYBENCH_2D(a,N,N,n,n), DATA_TYPE POLYBENCH_2D(a_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(b,N,N,n,n), DATA_TYPE POLYBENCH_2D(b_outputFromGpu,N,N,n,n))
{
	int i, j, fail;
	fail = 0;   

	// Compare output from CPU and GPU
	for (i=0; i<n; i++) 
	{
		for (j=0; j<n; j++) 
		{
			if (percentDiff(a[i][j], a_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
        }
	}
  
	for (i=0; i<n; i++) 
	{
       	for (j=0; j<n; j++) 
		{
        		if (percentDiff(b[i][j], b_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
        			fail++;
        		}
       	}
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void runJacobi2DCUDA(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(B_outputFromGpu,N,N,n,n))
{
	DATA_TYPE* Agpu;
	DATA_TYPE* Bgpu;
	
	// Create CUDA streams for asynchronous operations
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	cudaMalloc(&Agpu, N * N * sizeof(DATA_TYPE));
	cudaMalloc(&Bgpu, N * N * sizeof(DATA_TYPE));
	
	// Use async memcpy with stream1 for A and stream2 for B
	cudaMemcpyAsync(Agpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(Bgpu, B, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice, stream2);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), (unsigned int)ceil( ((float)N) / ((float)block.y) ));
	
	// Synchronize streams to ensure data is copied before kernel launch
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);
	
	/* Start timer. */
  	polybench_start_instruments;

	for (int t = 0; t < _PB_TSTEPS; t++)
	{
		runJacobiCUDA_kernel1<<<grid,block,0,stream1>>>(n, Agpu, Bgpu);
		cudaStreamSynchronize(stream1);
		runJacobiCUDA_kernel2<<<grid,block,0,stream2>>>(n, Agpu, Bgpu);
		cudaStreamSynchronize(stream2);
	}

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
  	polybench_print_instruments;
	
	// Use async memcpy for copying results back
	cudaMemcpyAsync(A_outputFromGpu, Agpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(B_outputFromGpu, Bgpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost, stream2);
	
	// Synchronize before freeing memory
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	cudaFree(Agpu);
	cudaFree(Bgpu);
	
	// Destroy streams
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char** argv)
{
	/* Retrieve problem size. */
	int n = N;
	int tsteps = TSTEPS;

	POLYBENCH_2D_ARRAY_DECL(a,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(b,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu,DATA_TYPE,N,N,n,n);

	init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
	runJacobi2DCUDA(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		runJacobi2DCpu(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	  	polybench_print_instruments;
	
		compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(n, POLYBENCH_ARRAY(a_outputFromGpu));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(a);
	POLYBENCH_FREE_ARRAY(a_outputFromGpu);
	POLYBENCH_FREE_ARRAY(b);
	POLYBENCH_FREE_ARRAY(b_outputFromGpu);

	return 0;
}

#include "../../common/polybench.c"

