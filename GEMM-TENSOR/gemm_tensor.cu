#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
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

#define ALPHA 1.7f
#define BETA 0.9f

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

	*alpha = ALPHA;
	*beta = BETA;

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


double tensorPercentDiff(double val1, double val2)
{
	if (fabs(val1) < 1e-10 && fabs(val2) < 1e-10) return 0.0;
	if (fabs(val1) < 1e-10 || fabs(val2) < 1e-10) return 100.0;
	return 100.0 * fabs((val1 - val2) / val1);
}


void compareResults(int ni, int nj, float* C, half* C_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	double max_diff = 0.0;
	double avg_diff = 0.0;
	int total_elements = ni * nj;
	int valid_comparisons = 0;
	int mismatch_count = 0;
	int max_i = -1, max_j = -1;
	float max_cpu_val = 0.0f, max_gpu_val = 0.0f;
	
	for (i=0; i < ni; i++) 
	{
		for (j=0; j < nj; j++) 
		{
			float cpu_val = __half2float(__float2half(C[i * nj + j]));
			float gpu_val = __half2float(C_outputFromGpu[i * nj + j]);
			double diff = tesnorPercentDiff(cpu_val, gpu_val);
			
			if (!isinf(diff) && !isnan(diff)) {
				avg_diff += diff;
				valid_comparisons++;
				if (diff > max_diff) {
					max_diff = diff;
					max_i = i;
					max_j = j;
					max_cpu_val = cpu_val;
					max_gpu_val = gpu_val;
				}
			}
			
			if (diff > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
				if (mismatch_count < 5) {
					printf("[Mismatch #%d] Position [%d,%d]: CPU=%.6f, GPU=%.6f, Diff=%.2f%%\n",
					       mismatch_count + 1, i, j, cpu_val, gpu_val, diff);
					mismatch_count++;
				}
			}
		}
	}
	
	if (max_i >= 0) {
		printf("[MAX Difference] Position [%d,%d]: CPU=%.6f, GPU=%.6f, Diff=%.2f%%\n",
		       max_i, max_j, max_cpu_val, max_gpu_val, max_diff);
	}
	
	if (valid_comparisons > 0) {
		avg_diff /= valid_comparisons;
	}
	
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d (%.2f%%)\n", 
	       PERCENT_DIFF_ERROR_THRESHOLD, fail, (100.0 * fail) / total_elements);
	printf("Average difference: %.4f%%, Max difference: %.4f%% (over %d valid comparisons)\n", 
	       avg_diff, max_diff, valid_comparisons);
}

__global__ void gemm_wmma_kernel(int M, int N, int K, half alpha, half beta,
                                  const half* A, const half* B, half* C)
{
	using namespace nvcuda::wmma;
	
	__shared__ half shared_A[BLOCK_SIZE_M][BLOCK_SIZE_K];
	__shared__ half shared_B[BLOCK_SIZE_K][BLOCK_SIZE_N];
	
	int warp_m = (threadIdx.x / warpSize);
	int warp_n = threadIdx.y;
	
	int block_m = blockIdx.x;
	int block_n = blockIdx.y;
	
	int global_warp_m = block_m * (BLOCK_SIZE_M / WMMA_M) + warp_m;
	int global_warp_n = block_n * (BLOCK_SIZE_N / WMMA_N) + warp_n;
	
	int row_start = global_warp_m * WMMA_M;
	int col_start = global_warp_n * WMMA_N;
	
	fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
	fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_b;
	fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc;
	fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_c;
	
	fill_fragment(acc, __float2half(0.0f));
	
	// Tile across K dimension
	for (int k_tile = 0; k_tile < K; k_tile += BLOCK_SIZE_K) {
		// Cooperative loading of A tile into shared memory
		int num_threads = blockDim.x * blockDim.y;
		int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
		int tile_size = BLOCK_SIZE_M * BLOCK_SIZE_K;
		
		for (int i = thread_id; i < tile_size; i += num_threads) {
			int row = i / BLOCK_SIZE_K;
			int col = i % BLOCK_SIZE_K;
			int global_row = block_m * BLOCK_SIZE_M + row;
			int global_col = k_tile + col;
			
			if (global_row < M && global_col < K) {
				shared_A[row][col] = A[global_row * K + global_col];
			} else {
				shared_A[row][col] = __float2half(0.0f);
			}
		}
		
		// Cooperative loading of B tile into shared memory
		tile_size = BLOCK_SIZE_K * BLOCK_SIZE_N;
		for (int i = thread_id; i < tile_size; i += num_threads) {
			int row = i / BLOCK_SIZE_N;
			int col = i % BLOCK_SIZE_N;
			int global_row = k_tile + row;
			int global_col = block_n * BLOCK_SIZE_N + col;
			
			if (global_row < K && global_col < N) {
				shared_B[row][col] = B[global_row * N + global_col];
			} else {
				shared_B[row][col] = __float2half(0.0f);
			}
		}
		
		__syncthreads();
		
		// Compute using shared memory
		for (int k_step = 0; k_step < BLOCK_SIZE_K; k_step += WMMA_K) {
			int smem_a_row = warp_m * WMMA_M;
			int smem_a_col = k_step;
			
			int smem_b_row = k_step;
			int smem_b_col = warp_n * WMMA_N;
			
			if (row_start < M && col_start < N && 
			    (k_tile + k_step) < K) {
				load_matrix_sync(frag_a, &shared_A[smem_a_row][smem_a_col], BLOCK_SIZE_K);
				load_matrix_sync(frag_b, &shared_B[smem_b_row][smem_b_col], BLOCK_SIZE_N);
				
				mma_sync(acc, frag_a, frag_b, acc);
			}
		}
		
		__syncthreads();
	}
	
	if (row_start < M && col_start < N) {
		load_matrix_sync(frag_c, C + row_start * N + col_start, N, mem_row_major);
		
		for (int idx = 0; idx < frag_c.num_elements; idx++) {
			frag_c.x[idx] = __hmul(frag_c.x[idx], beta);
		}
		
		for (int idx = 0; idx < acc.num_elements; idx++) {
			acc.x[idx] = __hmul(acc.x[idx], alpha);
			acc.x[idx] = __hadd(acc.x[idx], frag_c.x[idx]);
		}
		
		store_matrix_sync(C + row_start * N + col_start, acc, N, mem_row_major);
	}
}


void gemmCuda_Tensor(int ni, int nj, int nk, float alpha_f, float beta_f,
                     half* A, half* B, half* C, half* C_outputFromGpu)
{
	half *d_A, *d_B, *d_C;
	
	half alpha_h = __float2half(alpha_f);
	half beta_h = __float2half(beta_f);

	cudaMalloc((void **)&d_A, sizeof(half) * ni * nk);
	cudaMalloc((void **)&d_B, sizeof(half) * nk * nj);
	cudaMalloc((void **)&d_C, sizeof(half) * ni * nj);
	
	cudaMemcpy(d_A, A, sizeof(half) * ni * nk, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeof(half) * nk * nj, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, sizeof(half) * ni * nj, cudaMemcpyHostToDevice);
	
	dim3 block_dim(64, 2);
	dim3 grid_dim((ni + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
	              (nj + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);

  	polybench_start_instruments;

	gemm_wmma_kernel<<< grid_dim, block_dim >>>(ni, nj, nk, alpha_h, beta_h, d_A, d_B, d_C);
	cudaDeviceSynchronize();

	printf("GPU Tensor Core (WMMA FP16) Time in seconds:\n");
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