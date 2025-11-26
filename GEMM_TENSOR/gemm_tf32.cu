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


void init(int ni, int nj, int nk, float* alpha, float* beta, float* A, float* B, float* C)
{
	int i, j;

	*alpha = ALPHA;
	*beta = BETA;

  	for (i = 0; i < ni; i++)
	{
    		for (j = 0; j < nk; j++)
		{
      			A[i * nk + j] = ((float) i*j) / ni;
		}
	}

  	for (i = 0; i < nk; i++)
	{
    		for (j = 0; j < nj; j++)
		{
      			B[i * nj + j] = ((float) i*j) / ni;
		}
	}

  	for (i = 0; i < ni; i++)
	{
    		for (j = 0; j < nj; j++)
		{
      			C[i * nj + j] = ((float) i*j) / ni;
		}
	}
}


double tensorPercentDiff(double val1, double val2)
{
	if (fabs(val1) < 1e-10 && fabs(val2) < 1e-10) return 0.0;
	if (fabs(val1) < 1e-10 || fabs(val2) < 1e-10) return 100.0;
	return 100.0 * fabs((val1 - val2) / val1);
}


void compareResults(int ni, int nj, float* C, float* C_outputFromGpu)
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
			float cpu_val = C[i * nj + j];
			float gpu_val = C_outputFromGpu[i * nj + j];
			double diff = tensorPercentDiff(cpu_val, gpu_val);
			
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

__global__ void gemm_wmma_kernel(int M, int N, int K, float alpha, float beta,
                                  const float* A, const float* B, float* C)
{
	using namespace nvcuda::wmma;
	
	// Bank conflict optimization: Add padding for 16-byte alignment
	// For FP32: 16 floats = 64 bytes, +4 padding = 20 floats = 80 bytes (16-byte aligned)
	__shared__ float shared_A[BLOCK_SIZE_M][BLOCK_SIZE_K + 4];
	__shared__ float shared_B[BLOCK_SIZE_K][BLOCK_SIZE_N + 4];
	
	int warp_m = (threadIdx.x / warpSize);
	int warp_n = threadIdx.y;
	
	int block_m = blockIdx.x;
	int block_n = blockIdx.y;
	
	int global_warp_m = block_m * (BLOCK_SIZE_M / WMMA_M) + warp_m;
	int global_warp_n = block_n * (BLOCK_SIZE_N / WMMA_N) + warp_n;
	
	int row_start = global_warp_m * WMMA_M;
	int col_start = global_warp_n * WMMA_N;
	
	// TF32 tensor cores: 16x16x8 tiles (K=8 for TF32, not 16 like FP16)
	// precision::tf32 triggers HMMA.1684.F32.TF32 instructions
	fragment<matrix_a, WMMA_M, WMMA_N, 8, precision::tf32, row_major> frag_a;
	fragment<matrix_b, WMMA_M, WMMA_N, 8, precision::tf32, row_major> frag_b;
	fragment<accumulator, WMMA_M, WMMA_N, 8, float> acc;  // FP32 accumulator
	fragment<accumulator, WMMA_M, WMMA_N, 8, float> frag_c;
	
	fill_fragment(acc, 0.0f);
	
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
				shared_A[row][col] = 0.0f;
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
				shared_B[row][col] = 0.0f;
			}
		}
		
		__syncthreads();
		
		// Compute using shared memory with TF32 precision
		// K dimension is 8 for TF32 (not 16 like FP16)
		for (int k_step = 0; k_step < BLOCK_SIZE_K; k_step += 8) {
			int smem_a_row = warp_m * WMMA_M;
			int smem_a_col = k_step;
			
			int smem_b_row = k_step;
			int smem_b_col = warp_n * WMMA_N;
			
			if (row_start < M && col_start < N && 
			    (k_tile + k_step) < K) {
				load_matrix_sync(frag_a, &shared_A[smem_a_row][smem_a_col], BLOCK_SIZE_K + 4);
				load_matrix_sync(frag_b, &shared_B[smem_b_row][smem_b_col], BLOCK_SIZE_N + 4);
				
				mma_sync(acc, frag_a, frag_b, acc);  // HMMA.1684.F32.TF32 instruction
			}
		}
		
		__syncthreads();
	}
	
	// Epilogue: C = alpha * (A*B) + beta * C
	if (row_start < M && col_start < N) {
		// Load original C values as FP32
		load_matrix_sync(frag_c, C + row_start * N + col_start, N, mem_row_major);
		
		// Compute C = alpha * acc + beta * C in FP32
		#pragma unroll
		for (int idx = 0; idx < frag_c.num_elements; idx++) {
			acc.x[idx] = fmaf(acc.x[idx], alpha, frag_c.x[idx] * beta);
		}
		
		// Store result back to global memory
		store_matrix_sync(C + row_start * N + col_start, acc, N, mem_row_major);
	}
}


void gemmCuda_Tensor(int ni, int nj, int nk, float alpha, float beta,
                     float* A, float* B, float* C, float* C_outputFromGpu)
{
	float *d_A, *d_B, *d_C;

	cudaMalloc((void **)&d_A, sizeof(float) * ni * nk);
	cudaMalloc((void **)&d_B, sizeof(float) * nk * nj);
	cudaMalloc((void **)&d_C, sizeof(float) * ni * nj);
	
	// Enable TF32 tensor core operations (Ampere and later)
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("GPU: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);
	printf("TF32 Tensor Cores: %s\n", (prop.major >= 8) ? "Available ✓" : "Not Available");
	
	// Create CUDA stream for async operations
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	// Use cudaMemcpyAsync for non-blocking transfers
	cudaMemcpyAsync(d_A, A, sizeof(float) * ni * nk, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_B, B, sizeof(float) * nk * nj, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_C, C, sizeof(float) * ni * nj, cudaMemcpyHostToDevice, stream);
	
	dim3 block_dim(128, 4);  // 512 threads total = 16 warps to cover 64×64 tile
	dim3 grid_dim((ni + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
	              (nj + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);

  	polybench_start_instruments;

	gemm_wmma_kernel<<<grid_dim, block_dim, 0, stream>>>(ni, nj, nk, alpha, beta, d_A, d_B, d_C);
	cudaStreamSynchronize(stream);

	printf("GPU Tensor Core (WMMA TF32) Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpyAsync(C_outputFromGpu, d_C, sizeof(float) * ni * nj, cudaMemcpyDeviceToHost, stream);    
	cudaStreamSynchronize(stream);
	
	cudaStreamDestroy(stream);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

static
void print_array(int ni, int nj, float* C)
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, "%0.2f ", C[i * nj + j]);
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
	
	float* A = (float*)malloc(ni * nk * sizeof(float));
	float* B = (float*)malloc(nk * nj * sizeof(float));
	float* C = (float*)malloc(ni * nj * sizeof(float));
	float* C_outputFromGpu = (float*)malloc(ni * nj * sizeof(float));

	init(ni, nj, nk, &alpha, &beta, A, B, C);
	
	GPU_argv_init();
	
	gemmCuda_Tensor(ni, nj, nk, alpha, beta, A, B, C, C_outputFromGpu);

	#ifdef RUN_ON_CPU

	  	polybench_start_instruments;

		gemm(ni, nj, nk, alpha, beta, A, B, C);
		
		printf("CPU Time in seconds:\n");
  		polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(ni, nj, C, C_outputFromGpu);

	#else

		print_array(ni, nj, C_outputFromGpu);

	#endif


	free(A);
	free(B);  
	free(C);  
	free(C_outputFromGpu);

    	return 0;
}

#include "../../common/polybench.c"
