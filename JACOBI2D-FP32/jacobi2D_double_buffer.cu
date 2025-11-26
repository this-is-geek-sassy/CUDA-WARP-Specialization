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
#include <stdint.h>

#define POLYBENCH_TIME 1

#include "jacobi2D.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

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
	// Double buffering: Two shared memory tiles for overlapping load/compute
	__shared__ DATA_TYPE tile[2][DIM_THREAD_BLOCK_Y + 2][DIM_THREAD_BLOCK_X + 2];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = blockIdx.y * blockDim.y + ty;
	int j = blockIdx.x * blockDim.x + tx;
	
	int read_buffer = 0;  // Buffer to read from (can swap with write_buffer in future iterations)

	// Load initial data into buffer 0
	if (i < n && j < n)
	{
		tile[0][ty + 1][tx + 1] = A[i * N + j];
	}
	
	// Load halo cells for buffer 0
	if (ty == 0 && i > 0)
		tile[0][0][tx + 1] = A[(i - 1) * N + j];
	if (ty == blockDim.y - 1 && i < n - 1)
		tile[0][ty + 2][tx + 1] = A[(i + 1) * N + j];
	if (tx == 0 && j > 0)
		tile[0][ty + 1][0] = A[i * N + (j - 1)];
	if (tx == blockDim.x - 1 && j < n - 1)
		tile[0][ty + 1][tx + 2] = A[i * N + (j + 1)];
	
	__syncthreads();

	// Compute using buffer 0, result written to global memory
	if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
	{
		B[i*N + j] = 0.2f * (tile[read_buffer][ty + 1][tx + 1] + 
		                      tile[read_buffer][ty + 1][tx] + 
		                      tile[read_buffer][ty + 1][tx + 2] + 
		                      tile[read_buffer][ty + 2][tx + 1] + 
		                      tile[read_buffer][ty][tx + 1]);
	}
}


__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	// Double buffering for kernel2 as well
	__shared__ DATA_TYPE tile[2][DIM_THREAD_BLOCK_Y + 2][DIM_THREAD_BLOCK_X + 2];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = blockIdx.y * blockDim.y + ty;
	int j = blockIdx.x * blockDim.x + tx;
	
	int read_buffer = 0;

	// Load B data into buffer 0
	if (i < n && j < n)
	{
		tile[0][ty + 1][tx + 1] = B[i * N + j];
	}
	
	__syncthreads();
	
	// Copy from shared memory to A
	if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
	{
		A[i*N + j] = tile[read_buffer][ty + 1][tx + 1];
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

