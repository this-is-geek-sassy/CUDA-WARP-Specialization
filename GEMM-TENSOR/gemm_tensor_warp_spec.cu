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

// For FP16 tensor cores, compare at FP16 precision level
#define PERCENT_DIFF_ERROR_THRESHOLD 0.15

#define ALPHA 1.7f
#define BETA 0.9f

// Warp specialization configuration
#define NUM_LOAD_WARPS 4      // 4 warps dedicated to loading
#define NUM_COMPUTE_WARPS 12  // 12 warps dedicated to computing
#define TOTAL_WARPS 16        // Total warps per block

// Performance counters (in device memory)
__device__ unsigned long long d_load_count = 0;
__device__ unsigned long long d_compute_count = 0;

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

void gemm(int ni, int nj, int nk, float alpha, float beta, float* A, float* B, float* C)
{
	int i, j, k;

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

double safePercentDiff(float v1, float v2) {
	float abs_v1 = fabs(v1);
	float abs_v2 = fabs(v2);
	float abs_diff = fabs(v1 - v2);
	float avg = (abs_v1 + abs_v2) / 2.0f;
	
	if (avg < 1e-10f) {
		return (abs_diff < 1e-10f) ? 0.0 : 100.0;
	}
	
	return (abs_diff / avg) * 100.0;
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
			double diff = safePercentDiff(cpu_val, gpu_val);
			
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

// TRUE WARP SPECIALIZATION: Different warps have different roles
__global__ void gemm_wmma_warp_specialized(int M, int N, int K, half alpha, half beta,
                                            const half* A, const half* B, half* C)
{
	using namespace nvcuda::wmma;
	
	// Double buffering with warp specialization
	__shared__ half shared_A[2][BLOCK_SIZE_M][BLOCK_SIZE_K];
	__shared__ half shared_B[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
	__shared__ volatile int buffer_ready[2];  // Signals when buffer is loaded
	
	int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
	int lane_id = threadIdx.x % warpSize;
	
	int block_m = blockIdx.x;
	int block_n = blockIdx.y;
	
	// WARP SPECIALIZATION: Assign roles based on warp_id
	bool is_load_warp = (warp_id < NUM_LOAD_WARPS);
	
	// For compute warps: map to 3×4 grid (12 warps)
	// Since we only have 12 compute warps for 16 positions (4×4), use 3×4 layout
	int compute_warp_id = warp_id - NUM_LOAD_WARPS;
	int warp_m = compute_warp_id / 4;  // 0, 1, 2
	int warp_n = compute_warp_id % 4;  // 0, 1, 2, 3
	
	// Only compute warps calculate output positions
	int global_warp_m = block_m * (BLOCK_SIZE_M / WMMA_M) + warp_m;
	int global_warp_n = block_n * (BLOCK_SIZE_N / WMMA_N) + warp_n;
	
	int row_start = global_warp_m * WMMA_M;
	int col_start = global_warp_n * WMMA_N;
	
	// Only compute warps need fragments
	fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
	fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_b;
	fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
	
	if (!is_load_warp && compute_warp_id >= 0 && compute_warp_id < NUM_COMPUTE_WARPS) {
		fill_fragment(acc, 0.0f);
	}
	
	// Initialize buffer ready flags
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		buffer_ready[0] = 0;
		buffer_ready[1] = 0;
	}
	__syncthreads();
	
	int current_buffer = 0;
	int num_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
	
	// LOAD WARPS: Prefetch first tile
	if (is_load_warp) {
		int load_warp_id = warp_id;
		int elements_per_load_warp = (BLOCK_SIZE_M * BLOCK_SIZE_K) / NUM_LOAD_WARPS;
		int start_idx = load_warp_id * elements_per_load_warp;
		int end_idx = start_idx + elements_per_load_warp;
		
		// Load A tile into buffer 0
		for (int i = start_idx + lane_id; i < end_idx; i += warpSize) {
			int row = i / BLOCK_SIZE_K;
			int col = i % BLOCK_SIZE_K;
			int global_row = block_m * BLOCK_SIZE_M + row;
			int global_col = col;
			
			if (global_row < M && global_col < K) {
				shared_A[0][row][col] = A[global_row * K + global_col];
			} else {
				shared_A[0][row][col] = __float2half(0.0f);
			}
		}
		
		// Load B tile into buffer 0
		elements_per_load_warp = (BLOCK_SIZE_K * BLOCK_SIZE_N) / NUM_LOAD_WARPS;
		start_idx = load_warp_id * elements_per_load_warp;
		end_idx = start_idx + elements_per_load_warp;
		
		for (int i = start_idx + lane_id; i < end_idx; i += warpSize) {
			int row = i / BLOCK_SIZE_N;
			int col = i % BLOCK_SIZE_N;
			int global_row = row;
			int global_col = block_n * BLOCK_SIZE_N + col;
			
			if (global_row < K && global_col < N) {
				shared_B[0][row][col] = B[global_row * N + global_col];
			} else {
				shared_B[0][row][col] = __float2half(0.0f);
			}
		}
		
		// Signal buffer 0 is ready
		if (lane_id == 0 && load_warp_id == 0) {
			buffer_ready[0] = 1;
		}
	}
	
	// Main loop
	for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
		int k_tile = tile_idx * BLOCK_SIZE_K;
		
		// COMPUTE WARPS: Wait for current buffer to be ready, then compute
		if (!is_load_warp && compute_warp_id >= 0 && compute_warp_id < NUM_COMPUTE_WARPS) {
			// Wait for buffer to be loaded
			while (buffer_ready[current_buffer] == 0) { /* spin wait */ }
			
			// Compute using current buffer
			for (int k_step = 0; k_step < BLOCK_SIZE_K; k_step += WMMA_K) {
				int smem_a_row = warp_m * WMMA_M;
				int smem_a_col = k_step;
				
				int smem_b_row = k_step;
				int smem_b_col = warp_n * WMMA_N;
				
				if (row_start < M && col_start < N && (k_tile + k_step) < K) {
					load_matrix_sync(frag_a, &shared_A[current_buffer][smem_a_row][smem_a_col], BLOCK_SIZE_K);
					load_matrix_sync(frag_b, &shared_B[current_buffer][smem_b_row][smem_b_col], BLOCK_SIZE_N);
					
					mma_sync(acc, frag_a, frag_b, acc);
					
					// Count compute operations (only once per warp)
					if (lane_id == 0 && k_step == 0) {
						atomicAdd((unsigned long long*)&d_compute_count, 1ULL);
					}
				}
			}
		}
		
		// LOAD WARPS: Load next tile into alternate buffer
		if (is_load_warp && (tile_idx + 1) < num_tiles) {
			int next_k_tile = (tile_idx + 1) * BLOCK_SIZE_K;
			int next_buffer = 1 - current_buffer;
			
			// Mark next buffer as not ready
			if (lane_id == 0 && warp_id == 0) {
				buffer_ready[next_buffer] = 0;
			}
			__threadfence_block();
			
			int load_warp_id = warp_id;
			int elements_per_load_warp = (BLOCK_SIZE_M * BLOCK_SIZE_K) / NUM_LOAD_WARPS;
			int start_idx = load_warp_id * elements_per_load_warp;
			int end_idx = start_idx + elements_per_load_warp;
			
			// Load next A tile
			for (int i = start_idx + lane_id; i < end_idx; i += warpSize) {
				int row = i / BLOCK_SIZE_K;
				int col = i % BLOCK_SIZE_K;
				int global_row = block_m * BLOCK_SIZE_M + row;
				int global_col = next_k_tile + col;
				
				if (global_row < M && global_col < K) {
					shared_A[next_buffer][row][col] = A[global_row * K + global_col];
				} else {
					shared_A[next_buffer][row][col] = __float2half(0.0f);
				}
			}
			
			// Load next B tile
			elements_per_load_warp = (BLOCK_SIZE_K * BLOCK_SIZE_N) / NUM_LOAD_WARPS;
			start_idx = load_warp_id * elements_per_load_warp;
			end_idx = start_idx + elements_per_load_warp;
			
			for (int i = start_idx + lane_id; i < end_idx; i += warpSize) {
				int row = i / BLOCK_SIZE_N;
				int col = i % BLOCK_SIZE_N;
				int global_row = next_k_tile + row;
				int global_col = block_n * BLOCK_SIZE_N + col;
				
				if (global_row < K && global_col < N) {
					shared_B[next_buffer][row][col] = B[global_row * N + global_col];
				} else {
					shared_B[next_buffer][row][col] = __float2half(0.0f);
				}
			}
			
			// Signal next buffer is ready
			__threadfence_block();
			if (lane_id == 0 && load_warp_id == 0) {
				buffer_ready[next_buffer] = 1;
				atomicAdd((unsigned long long*)&d_load_count, 1ULL);
			}
		}
		
		__syncthreads();
		current_buffer = 1 - current_buffer;
	}
	
	// COMPUTE WARPS: Write results
	if (!is_load_warp && compute_warp_id >= 0 && compute_warp_id < NUM_COMPUTE_WARPS) {
		if (row_start < M && col_start < N) {
			fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_c_half;
			load_matrix_sync(frag_c_half, C + row_start * N + col_start, N, mem_row_major);
			
			fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
			#pragma unroll
			for (int idx = 0; idx < frag_c_half.num_elements; idx++) {
				frag_c.x[idx] = __half2float(frag_c_half.x[idx]);
			}
			
			float alpha_f = __half2float(alpha);
			float beta_f = __half2float(beta);
			
			#pragma unroll
			for (int idx = 0; idx < acc.num_elements; idx++) {
				acc.x[idx] = fmaf(acc.x[idx], alpha_f, frag_c.x[idx] * beta_f);
			}
			
			fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_half;
			#pragma unroll
			for (int idx = 0; idx < acc.num_elements; idx++) {
				acc_half.x[idx] = __float2half(acc.x[idx]);
			}
			
			store_matrix_sync(C + row_start * N + col_start, acc_half, N, mem_row_major);
		}
	}
}

void gemmCuda_Tensor_WarpSpec(int ni, int nj, int nk, float alpha_f, float beta_f,
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
	
	// Reset counters
	unsigned long long zero = 0;
	cudaMemcpyToSymbol(d_load_count, &zero, sizeof(unsigned long long));
	cudaMemcpyToSymbol(d_compute_count, &zero, sizeof(unsigned long long));
	
	dim3 block_dim(128, 4);  // 512 threads = 16 warps
	dim3 grid_dim((ni + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
	              (nj + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);

  	polybench_start_instruments;

	gemm_wmma_warp_specialized<<< grid_dim, block_dim >>>(ni, nj, nk, alpha_h, beta_h, d_A, d_B, d_C);
	cudaDeviceSynchronize();

	printf("GPU Tensor Core (WMMA FP16 + Warp Specialization) Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
	
	// Retrieve and print counters
	unsigned long long h_load_count, h_compute_count;
	cudaMemcpyFromSymbol(&h_load_count, d_load_count, sizeof(unsigned long long));
	cudaMemcpyFromSymbol(&h_compute_count, d_compute_count, sizeof(unsigned long long));
	
	int num_blocks = grid_dim.x * grid_dim.y;
	int num_tiles = (nk + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
	
	printf("\n=== WARP SPECIALIZATION VERIFICATION ===\n");
	printf("Load operations:    %llu (expected: %d blocks × %d tiles = %d)\n", 
	       h_load_count, num_blocks, num_tiles-1, num_blocks * (num_tiles-1));
	printf("Compute operations: %llu (expected: %d blocks × %d warps × %d tiles = %d)\n",
	       h_compute_count, num_blocks, NUM_COMPUTE_WARPS, num_tiles, 
	       num_blocks * NUM_COMPUTE_WARPS * num_tiles);
	printf("Load warps:         %d (first %d warps per block)\n", NUM_LOAD_WARPS, NUM_LOAD_WARPS);
	printf("Compute warps:      %d (warps %d-%d per block)\n", NUM_COMPUTE_WARPS, NUM_LOAD_WARPS, TOTAL_WARPS-1);
	printf("=========================================\n\n");

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
	
	gemmCuda_Tensor_WarpSpec(ni, nj, nk, alpha, beta, A, B, C, C_outputFromGpu);

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
