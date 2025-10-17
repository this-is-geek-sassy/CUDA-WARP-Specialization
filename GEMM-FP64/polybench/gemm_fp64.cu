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

// using namespace nvcuda;

#define POLYBENCH_TIME 1

#include "gemm_fp64.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

#include "../gpu_utils.h"

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define ALPHA 32412.0f
#define BETA 2123.0f

#define RUN_ON_CPU

void gemm(int ni, int nj, int nk, double alpha, double beta, double* A, double* B, double* C)
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


void init(int ni, int nj, int nk, double* alpha, double* beta, double* A, double* B, double* C)
{
	int i, j;

	*alpha = 32412;
	*beta = 2123;

  	for (i = 0; i < ni; i++)
	{
    		for (j = 0; j < nk; j++)
		{
      			A[i * nk + j] = (((double) i*j) / ni);
		}
	}

  	for (i = 0; i < nk; i++)
	{
    		for (j = 0; j < nj; j++)
		{
      			B[i * nj + j] = (((double) i*j) / ni);
		}
	}

  	for (i = 0; i < ni; i++)
	{
    		for (j = 0; j < nj; j++)
		{
      			C[i * nj + j] = (((double) i*j) / ni);
		}
	}
}


void compareResults(int ni, int nj, double* C, double* C_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < ni; i++) 
	{
		for (j=0; j < nj; j++) 
		{
			double gpu_val = C_outputFromGpu[i * nj + j];
			if (percentDiff(C[i * nj + j], gpu_val) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


template<unsigned int BM, unsigned int BN, unsigned int NUM_THREADS>
__device__ void loadTileChunked(const unsigned int N, double* src, double* dest) {
  float4* src_float4 = reinterpret_cast<float4*>(src);
  float4* dest_float4 = reinterpret_cast<float4*>(dest);
  constexpr unsigned int BN_VECTORIZED = BN / 2;
  const unsigned int N_vectorized = N / 2;

  static_assert(NUM_THREADS % BN_VECTORIZED == 0);
  constexpr unsigned int ROW_STEP = NUM_THREADS / BN_VECTORIZED; 
  constexpr unsigned int NUM_ITERS = BM / ROW_STEP;

  const unsigned int tId = threadIdx.y * blockDim.y + threadIdx.x;
  unsigned int row = tId / BN_VECTORIZED;
  const unsigned int col = tId % BN_VECTORIZED;

  #pragma unroll
  for(unsigned int i = 0; i < NUM_ITERS; i++) {
    dest_float4[row * BN_VECTORIZED + col] = src_float4[row * N_vectorized + col];
    row += ROW_STEP;
  }
}

template<unsigned int BM, unsigned int BK, unsigned int BN, unsigned int TM, unsigned int TN, unsigned int TK, unsigned int NUM_THREADS>
__global__ void dgemm_fp64(double alpha, double beta, int M, int N, int K, double* A, double* B, double* C) {
  extern __shared__ double sm[];
  double* sA = &sm[0];
  double* sB = &sm[BM * BK];

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  unsigned int bm = blockIdx.y * BM;
  unsigned int bn = blockIdx.x * BN;

  double a_reg[TM][TK];
  double b_reg[TK][TN];
  double acc_reg[TM][TN];
  for(int i = 0; i < TM; i++) 
    for(int j = 0; j < TN; j++)
      acc_reg[i][j] = 0.0;

  for(unsigned int bk = 0; bk < K; bk += BK) {
    double* gA = A + (bm * K + bk);
    double* gB = B + (bk * N + bn);
    loadTileChunked<BM, BK, NUM_THREADS>(K, gA, sA);
    loadTileChunked<BK, BN, NUM_THREADS>(N, gB, sB);
    __syncthreads();

    for(int wk = 0; wk < BK; wk += TK) {
      // Tiled loads into Register Memory (Need to check PTX and SASS to confirm unrolling and chunking)
      for(int k = 0; k < TK; k++) {
        for(int i = 0; i < TM; i++) a_reg[i][k] = sA[(ty * TM + i) * BK + wk + k];
        for(int j = 0; j < TN; j++) b_reg[k][j] = sB[(wk + k) * BN + tx * TN + j];
      }
  
      // FMA operations on Register Memory (Need to check PTX and SASS to confirm unrolling)
      for(int i = 0; i < TM; i++)
        for(int j = 0; j < TN; j++)
          for(int k = 0; k < TK; k++)
            acc_reg[i][j] = fma(a_reg[i][k], b_reg[k][j], acc_reg[i][j]);
    }
    __syncthreads();
  }

  for(int i = 0; i < TM; i++) {
    for(int j = 0; j < TN; j++) {
      C[(bm + ty * TM + i) * N + (bn + tx * TN + j)] = alpha * acc_reg[i][j] + beta * C[(bm + ty * TM + i) * N + (bn + tx * TN + j)];
    }
  }
}

void dgemm_driver(int M, int N, int K, double alpha, double beta, double* hA, double* hB, double* hC, double* C_outputFromGpu) {
  const unsigned int BM = 64;
  const unsigned int BK = 16;
  const unsigned int BN = 64;
  const unsigned int TM = 4;
  const unsigned int TN = 4;
  const unsigned int TK = 2;
  const unsigned int NUM_THREADS = (BN/TN) * (BM/TM);

  dim3 gridDim(N/BN, M/BM, 1);
  dim3 blockDim(BN/TN, BM/TM, 1);
  const size_t sharedMemSize = BK * (BM + BN) * sizeof(double);

  double *dA, *dB, *dC;
  cudaMalloc(&dA, M * K * sizeof(double));
  cudaMalloc(&dB, K * N * sizeof(double));
  cudaMalloc(&dC, M * N * sizeof(double));

  cudaMemcpy(dA, hA, M * K * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, K * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC, M * N * sizeof(double), cudaMemcpyHostToDevice);

  polybench_start_instruments;

  dgemm_fp64<BM, BK, BN, TM, TN, TK, NUM_THREADS><<<gridDim, blockDim, sharedMemSize>>>(alpha, beta, M, N, K, dA, dB, dC);
  cudaDeviceSynchronize();

  printf("GPU FP64 Core Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

  cudaMemcpy(C_outputFromGpu, dC, M * N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

static
void print_array(int ni, int nj, double* C)
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, "%f ", C[i * nj + j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	int ni = NI;
	int nj = NJ;
	int nk = NK;

	double alpha;
	double beta;
	
	double* A = (double*)malloc(ni * nk * sizeof(double));
	double* B = (double*)malloc(nk * nj * sizeof(double));
	double* C = (double*)malloc(ni * nj * sizeof(double));
	double* C_outputFromGpu = (double*)malloc(ni * nj * sizeof(double));

	init(ni, nj, nk, &alpha, &beta, A, B, C);
	GPU_argv_init();
	dgemm_driver(ni, nj, nk, alpha, beta, A, B, C, C_outputFromGpu);

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