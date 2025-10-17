#ifndef GLOBAL_MEM_UTILS_CUH
#define GLOBAL_MEM_UTILS_CUH

#include <cuda.h>
#include <cassert>

/// @brief Loads tile of dimension (BM x BN) from src into dst.
///        [Batches read operations through loop unrolling and instruction re-ordering]
/// @param NUM_THREADS Number of threads per block (compile time constant)
/// @param src Pointer to src
/// @param dest Pointer to dest
/// @param N Row stride of src
template<unsigned int BM, unsigned int BN, unsigned int NUM_THREADS>
__device__ void loadTileBatched(const unsigned int N, double* src, double* dest) {
  static_assert(NUM_THREADS % BN == 0);
  constexpr unsigned int ROW_STEP = NUM_THREADS / BN; 
  constexpr unsigned int NUM_ITERS = BM / ROW_STEP;

  const unsigned int tId = threadIdx.y * blockDim.y + threadIdx.x;
  unsigned int row = tId / BN;
  const unsigned int col = tId % BN;

  #pragma unroll
  for(unsigned int i = 0; i < NUM_ITERS; i++) {
    dest[row * BN + col] = src[row * N + col];
    row += ROW_STEP;
  }
}

/// @brief Loads tile of dimension (BM x BN) from src into dst.
///        [Batches read operations through loop unrolling and instruction re-ordering]
///        [Chunks 2 doubles together to have 128b load instructions]
/// @param NUM_THREADS Number of threads per block (compile time constant)
/// @param src Pointer to src
/// @param dest Pointer to dest
/// @param N Row stride of src
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

#endif // GLOBAL_MEM_UTILS_CUH
