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
#include "../common/polybench.h"
#include "../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

#include "../gpu_utils.h"

// For FP16 tensor cores, compare at FP16 precision level
#define PERCENT_DIFF_ERROR_THRESHOLD 0.10

#define ALPHA 1.7f
#define BETA 0.9f

#define RUN_ON_CPU

// NUM_LOAD_WARPS and NUM_COMPUTE_WARPS now defined in gemm_tensor.cuh

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


double safePercentDiff(double val1, double val2)
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

// WARP SPECIALIZED KERNEL: Separate load and compute warps with double buffering
__global__ void gemm_wmma_kernel(int M, int N, int K, half alpha, half beta,
                                  const half* A, const half* B, half* C)
{
	using namespace nvcuda::wmma;
	
	// Bank conflict optimization: Add padding (must be multiple of 8 for 16-byte alignment)
	__shared__ half shared_A[2][BLOCK_SIZE_M][BLOCK_SIZE_K + 8];
	__shared__ half shared_B[2][BLOCK_SIZE_K][BLOCK_SIZE_N + 8];
	
	int warp_id = (threadIdx.y * (blockDim.x / warpSize)) + (threadIdx.x / warpSize);
	int lane_id = threadIdx.x % warpSize;
	bool is_load_warp = (warp_id < NUM_LOAD_WARPS);
	
	int block_m = blockIdx.x;
	int block_n = blockIdx.y;
	int num_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
	
	if (is_load_warp) {
		int num_load_threads = NUM_LOAD_WARPS * warpSize;
		int load_thread_id = warp_id * warpSize + lane_id;
		
		for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
			int k_tile = tile_idx * BLOCK_SIZE_K;
			int buffer_idx = tile_idx % 2;
			
			// Load A tile using cp.async (4 bytes = 2 halfs at a time)
			int tile_size_A = BLOCK_SIZE_M * BLOCK_SIZE_K;
			for (int i = load_thread_id * 2; i < tile_size_A; i += num_load_threads * 2) {
				int row = i / BLOCK_SIZE_K;
				int col = i % BLOCK_SIZE_K;
				int global_row = block_m * BLOCK_SIZE_M + row;
				int global_col = k_tile + col;
				
				// Load 2 halfs (4 bytes) at once if both are in bounds
				if (global_row < M && global_col < K && (col + 1) < BLOCK_SIZE_K) {
					uint32_t smem_addr = __cvta_generic_to_shared(&shared_A[buffer_idx][row][col]);
					const half* src = &A[global_row * K + global_col];
					
					asm volatile(
						"cp.async.ca.shared.global [%0], [%1], 4;\n"
						:: "r"(smem_addr), "l"(src)
					);
				} else {
					// Handle boundary cases with regular loads
					if (global_row < M && global_col < K) {
						shared_A[buffer_idx][row][col] = A[global_row * K + global_col];
					} else {
						shared_A[buffer_idx][row][col] = __float2half(0.0f);
					}
					if (col + 1 < BLOCK_SIZE_K) {
						if (global_row < M && global_col + 1 < K) {
							shared_A[buffer_idx][row][col + 1] = A[global_row * K + global_col + 1];
						} else {
							shared_A[buffer_idx][row][col + 1] = __float2half(0.0f);
						}
					}
				}
			}
			
			// Load B tile using cp.async (4 bytes = 2 halfs)
			int tile_size_B = BLOCK_SIZE_K * BLOCK_SIZE_N;
			for (int i = load_thread_id * 2; i < tile_size_B; i += num_load_threads * 2) {
				int row = i / BLOCK_SIZE_N;
				int col = i % BLOCK_SIZE_N;
				int global_row = k_tile + row;
				int global_col = block_n * BLOCK_SIZE_N + col;
				
				// Load 2 halfs (4 bytes) at once
				if (global_row < K && global_col < N && (col + 1) < BLOCK_SIZE_N) {
					uint32_t smem_addr = __cvta_generic_to_shared(&shared_B[buffer_idx][row][col]);
					const half* src = &B[global_row * N + global_col];
					
					asm volatile(
						"cp.async.ca.shared.global [%0], [%1], 4;\n"
						:: "r"(smem_addr), "l"(src)
					);
				} else {
					// Handle boundary cases
					if (global_row < K && global_col < N) {
						shared_B[buffer_idx][row][col] = B[global_row * N + global_col];
					} else {
						shared_B[buffer_idx][row][col] = __float2half(0.0f);
					}
					if (col + 1 < BLOCK_SIZE_N) {
						if (global_row < K && global_col + 1 < N) {
							shared_B[buffer_idx][row][col + 1] = B[global_row * N + global_col + 1];
						} else {
							shared_B[buffer_idx][row][col + 1] = __float2half(0.0f);
						}
					}
				}
			}
			
			// Commit and wait for all cp.async operations
			asm volatile("cp.async.commit_group;\n");
			asm volatile("cp.async.wait_group 0;\n");
			
			// Sync: load warps done loading, compute warps can start
			__syncthreads();
		}
		
	} else {


		int compute_warp_local_id = warp_id - NUM_LOAD_WARPS;  // 0 to (NUM_COMPUTE_WARPS-1)
		int total_positions = BLOCK_TILES_M * BLOCK_TILES_N;  // 8×4 = 32 tiles for 128x64 block
		int positions_per_warp;
		int start_pos;
		
		int base_tiles = total_positions / NUM_COMPUTE_WARPS;  // 32/12 = 2
		int extra_tiles = total_positions % NUM_COMPUTE_WARPS;  // 32%12 = 8
		
		if (compute_warp_local_id < extra_tiles) {
			// First 'extra_tiles' warps get one more tile
			positions_per_warp = base_tiles + 1;
			start_pos = compute_warp_local_id * positions_per_warp;
		} else {
			// Remaining warps get base number of tiles
			positions_per_warp = base_tiles;
			start_pos = extra_tiles * (base_tiles + 1) + 
			            (compute_warp_local_id - extra_tiles) * base_tiles;
		}
		
		fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
		fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_b;
		fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[3];  // Max 3 tiles per warp (e.g., 6 compute warps)
		fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
		
		for (int p = 0; p < positions_per_warp; p++) {
			fill_fragment(acc[p], 0.0f);
		}
		
		for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
			int k_tile = tile_idx * BLOCK_SIZE_K;
			int buffer_idx = tile_idx % 2;
			
			// Sync: wait for load warps to finish loading
			__syncthreads();
			
			// Compute with current buffer - each warp handles 2-3 positions
			for (int p = 0; p < positions_per_warp; p++) {
				int pos = start_pos + p;
				int warp_m = pos / BLOCK_TILES_N;  // Row in 8×4 grid
				int warp_n = pos % BLOCK_TILES_N;  // Col in 8×4 grid
				
				int global_warp_m = block_m * (BLOCK_SIZE_M / WMMA_M) + warp_m;
				int global_warp_n = block_n * (BLOCK_SIZE_N / WMMA_N) + warp_n;
				
				int row_start = global_warp_m * WMMA_M;
				int col_start = global_warp_n * WMMA_N;
				
				for (int k_step = 0; k_step < BLOCK_SIZE_K; k_step += WMMA_K) {
					int smem_a_row = warp_m * WMMA_M;
					int smem_a_col = k_step;
					int smem_b_row = k_step;
					int smem_b_col = warp_n * WMMA_N;
					
					if (row_start < M && col_start < N && (k_tile + k_step) < K) {
						load_matrix_sync(frag_a, &shared_A[buffer_idx][smem_a_row][smem_a_col], BLOCK_SIZE_K + 8);
						load_matrix_sync(frag_b, &shared_B[buffer_idx][smem_b_row][smem_b_col], BLOCK_SIZE_N + 8);
						mma_sync(acc[p], frag_a, frag_b, acc[p]);
					}
				}
			}
		}
		
		// Write results - each warp writes its tile(s)
		for (int p = 0; p < positions_per_warp; p++) {
			int pos = start_pos + p;
			int warp_m = pos / BLOCK_TILES_N;
			int warp_n = pos % BLOCK_TILES_N;
			
			int global_warp_m = block_m * (BLOCK_SIZE_M / WMMA_M) + warp_m;
			int global_warp_n = block_n * (BLOCK_SIZE_N / WMMA_N) + warp_n;
			
			int row_start = global_warp_m * WMMA_M;
			int col_start = global_warp_n * WMMA_N;
			
			if (row_start < M && col_start < N) {
				fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_c_half;
				load_matrix_sync(frag_c_half, C + row_start * N + col_start, N, mem_row_major);
				
				#pragma unroll
				for (int idx = 0; idx < frag_c_half.num_elements; idx++) {
					frag_c.x[idx] = __half2float(frag_c_half.x[idx]);
				}
				
				float alpha_f = __half2float(alpha);
				float beta_f = __half2float(beta);
				
				#pragma unroll
				for (int idx = 0; idx < acc[p].num_elements; idx++) {
					acc[p].x[idx] = fmaf(acc[p].x[idx], alpha_f, frag_c.x[idx] * beta_f);
				}
				
				fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_half;
				#pragma unroll
				for (int idx = 0; idx < acc[p].num_elements; idx++) {
					acc_half.x[idx] = __float2half(acc[p].x[idx]);
				}
				
				store_matrix_sync(C + row_start * N + col_start, acc_half, N, mem_row_major);
			}
		}
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
	
	// Create CUDA stream for async operations
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	// Use cudaMemcpyAsync for non-blocking transfers
	cudaMemcpyAsync(d_A, A, sizeof(half) * ni * nk, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_B, B, sizeof(half) * nk * nj, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_C, C, sizeof(half) * ni * nj, cudaMemcpyHostToDevice, stream);
	
	// For 128x64 rectangular tile with 16 warps (512 threads)
	dim3 block_dim(64, 8);  // 64 threads x 8 rows = 512 threads = 16 warps
	dim3 grid_dim((ni + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
	              (nj + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);

  	polybench_start_instruments;

	gemm_wmma_kernel<<<grid_dim, block_dim, 0, stream>>>(ni, nj, nk, alpha_h, beta_h, d_A, d_B, d_C);
	cudaStreamSynchronize(stream);

	printf("GPU Tensor Core (WMMA FP16) Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpyAsync(C_outputFromGpu, d_C, sizeof(half) * ni * nj, cudaMemcpyDeviceToHost, stream);    
	cudaStreamSynchronize(stream);
	
	cudaStreamDestroy(stream);
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

#include "../common/polybench.c"