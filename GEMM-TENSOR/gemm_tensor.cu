#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define POLYBENCH_TIME 1

#include "gemm_tensor.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

#include "../gpu_utils.h"

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define ALPHA 32412.0f
#define BETA 2123.0f

#define RUN_ON_CPU


void gemm(int ni, int nj, int nk, float alpha, float beta, float* A, float* B, float* C)
{
	int i,j,k;
	
	for (i = 0; i < ni; i++)
	{
    		for (j = 0; j < nj; j++)
    		{
			C[i * nj + j] *= beta;
	
			for (k = 0; k < nk; ++k)
			{
	  			C[i * nj + j] += alpha * A[i * nk + k] * B[k * nj + j];
			}
      		}
	}
}


void init(int ni, int nj, int nk, float* alpha, float* beta, half* A, half* B, half* C)
{
	int i, j;

	*alpha = 32412;
	*beta = 2123;

  	for (i = 0; i < ni; i++)
	{
    		for (j = 0; j < nk; j++)
		{
      			A[i * nk + j] = __float2half(((float) i*j) / ni);
		}
	}

  	for (i = 0; i < nk; i++)
	{
    		for (j = 0; j < nj; j++)
		{
      			B[i * nj + j] = __float2half(((float) i*j) / ni);
		}
	}

  	for (i = 0; i < ni; i++)
	{
    		for (j = 0; j < nj; j++)
		{
      			C[i * nj + j] = __float2half(((float) i*j) / ni);
		}
	}
}


void compareResults(int ni, int nj, float* C, half* C_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < ni; i++) 
	{
		for (j=0; j < nj; j++) 
		{
			float gpu_val = __half2float(C_outputFromGpu[i * nj + j]);
			if (percentDiff(C[i * nj + j], gpu_val) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

__global__ void gemm_wmma_kernel(int M, int N, int K, half alpha, half beta,
                                  const half* A, const half* B, half* C)
{
	int warp_id_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	int warp_id_n = (blockIdx.y * blockDim.y + threadIdx.y);
	
	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> accumulator;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_c;
	
	wmma::fill_fragment(accumulator, __float2half(0.0f));
	
	int row_start = warp_id_m * WMMA_M;
	int col_start = warp_id_n * WMMA_N;
	
	if (row_start >= M || col_start >= N) {
		return;
	}
	
	for (int k_step = 0; k_step < K; k_step += WMMA_K) {
		int a_row = row_start;
		int a_col = k_step;
		
		int b_row = k_step;
		int b_col = col_start;
		
		if (a_row < M && a_col < K && b_row < K && b_col < N) {
			wmma::load_matrix_sync(frag_a, A + a_row * K + a_col, K);
			wmma::load_matrix_sync(frag_b, B + b_row * N + b_col, N);
			
			wmma::mma_sync(accumulator, frag_a, frag_b, accumulator);
		}
	}
	
	if (row_start < M && col_start < N) {
		wmma::load_matrix_sync(frag_c, C + row_start * N + col_start, N, wmma::mem_row_major);
		
		for (int idx = 0; idx < frag_c.num_elements; idx++) {
			frag_c.x[idx] = __hmul(frag_c.x[idx], beta);
		}
		
		for (int idx = 0; idx < accumulator.num_elements; idx++) {
			accumulator.x[idx] = __hmul(accumulator.x[idx], alpha);
			accumulator.x[idx] = __hadd(accumulator.x[idx], frag_c.x[idx]);
		}
		
		wmma::store_matrix_sync(C + row_start * N + col_start, accumulator, N, wmma::mem_row_major);
	}
}


void gemmCuda_Tensor(int ni, int nj, int nk, float alpha_f, float beta_f,
                     half* A, half* B, half* C, half* C_outputFromGpu)
{
	half *d_A;
	half *d_B;
	half *d_C;
	
	half alpha_h = __float2half(alpha_f);
	half beta_h = __float2half(beta_f);

	cudaMalloc((void **)&d_A, sizeof(half) * ni * nk);
	cudaMalloc((void **)&d_B, sizeof(half) * nk * nj);
	cudaMalloc((void **)&d_C, sizeof(half) * ni * nj);
	
	cudaMemcpy(d_A, A, sizeof(half) * ni * nk, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeof(half) * nk * nj, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, sizeof(half) * ni * nj, cudaMemcpyHostToDevice);
	
	dim3 block_dim(32, 8);
	dim3 grid_dim((ni + WMMA_M * block_dim.x / 32 - 1) / (WMMA_M * block_dim.x / 32),
	              (nj + WMMA_N * block_dim.y - 1) / (WMMA_N * block_dim.y));

  	polybench_start_instruments;

	gemm_wmma_kernel<<< grid_dim, block_dim >>>(ni, nj, nk, alpha_h, beta_h, d_A, d_B, d_C);
	cudaDeviceSynchronize();

	printf("GPU Tensor Core Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(C_outputFromGpu, d_C, sizeof(half) * ni * nj, cudaMemcpyDeviceToHost);    
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

static
void print_array(int ni, int nj, half* C)
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, "%0.2f ", __half2float(C[i * nj + j]));
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	int ni = NI;
	int nj = NJ;
	int nk = NK;

	float alpha;
	float beta;
	
	half* A = (half*)malloc(ni * nk * sizeof(half));
	half* B = (half*)malloc(nk * nj * sizeof(half));
	half* C = (half*)malloc(ni * nj * sizeof(half));
	half* C_outputFromGpu = (half*)malloc(ni * nj * sizeof(half));
	
	float* A_float = (float*)malloc(ni * nk * sizeof(float));
	float* B_float = (float*)malloc(nk * nj * sizeof(float));
	float* C_float = (float*)malloc(ni * nj * sizeof(float));

	init(ni, nj, nk, &alpha, &beta, A, B, C);
	
	for (int idx = 0; idx < ni * nk; idx++) A_float[idx] = __half2float(A[idx]);
	for (int idx = 0; idx < nk * nj; idx++) B_float[idx] = __half2float(B[idx]);
	for (int idx = 0; idx < ni * nj; idx++) C_float[idx] = __half2float(C[idx]);
	
	GPU_argv_init();
	
	gemmCuda_Tensor(ni, nj, nk, alpha, beta, A, B, C, C_outputFromGpu);

	#ifdef RUN_ON_CPU

	  	polybench_start_instruments;

		gemm(ni, nj, nk, alpha, beta, A_float, B_float, C_float);
		
		printf("CPU Time in seconds:\n");
  		polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(ni, nj, C_float, C_outputFromGpu);

	#else

		print_array(ni, nj, C_outputFromGpu);

	#endif


	free(A);
	free(B);  
	free(C);  
	free(C_outputFromGpu);
	free(A_float);
	free(B_float);
	free(C_float);

    	return 0;
}

#include "../../common/polybench.c"
