#ifndef GLOBAL_MEM_UTILS_CUH
#define GLOBAL_MEM_UTILS_CUH

#include <stdio.h>
#include <cuda.h>
#include <cassert>
#include <cuda/barrier>

/// @brief Reads tile of dimension (BM x BN) from src into dst.
///        [Batches read operations through loop unrolling and instruction re-ordering]
/// @param NUM_THREADS Number of threads per block (compile time constant)
/// @param src Pointer to src
/// @param dest Pointer to dest
/// @param N Row stride of src
template<unsigned int BM, unsigned int BN, unsigned int NUM_THREADS, unsigned int THREAD_OFFSET>
__device__ void readTileBatched(const unsigned int N, float* src, float* dest) {
  static_assert(NUM_THREADS % BN == 0);
  constexpr unsigned int ROW_STEP = NUM_THREADS / BN;
  static_assert(BM % ROW_STEP == 0);
  constexpr unsigned int NUM_ITERS = BM / ROW_STEP;

  const unsigned int tId = threadIdx.y * blockDim.x + threadIdx.x - THREAD_OFFSET;
  unsigned int row = tId / BN;
  const unsigned int col = tId % BN;

  #pragma unroll
  for(unsigned int i = 0; i < NUM_ITERS; i++) {
    dest[row * BN + col] = src[row * N + col];
    row += ROW_STEP;
  }
}

/// @brief Reads tile of dimension (BM x BN) from src into dst.
///        [Batches read operations through loop unrolling and instruction re-ordering]
///        [Chunks 2 doubles together to have 128b load instructions]
/// @param NUM_THREADS Number of threads per block (compile time constant)
/// @param src Pointer to src
/// @param dest Pointer to dest
/// @param N Row stride of src
template<unsigned int BM, unsigned int BN, unsigned int NUM_THREADS, unsigned int THREAD_OFFSET>
__forceinline__ __device__ void readTileChunked(const unsigned int N, float* src, float* dest) {
  float4* src_float4 = reinterpret_cast<float4*>(src);
  float4* dest_float4 = reinterpret_cast<float4*>(dest);
  constexpr unsigned int BN_VECTORIZED = BN / 4;
  const unsigned int N_vectorized = N / 4;

  static_assert(NUM_THREADS % BN_VECTORIZED == 0);
  constexpr unsigned int ROW_STEP = NUM_THREADS / BN_VECTORIZED; 
  // static_assert(BM % ROW_STEP == 0);
  constexpr unsigned int NUM_ITERS = (BM + ROW_STEP - 1) / ROW_STEP;

  const unsigned int tId = threadIdx.y * blockDim.x + threadIdx.x - THREAD_OFFSET;
  unsigned int row = tId / BN_VECTORIZED;
  const unsigned int col = tId % BN_VECTORIZED;

  #pragma unroll
  for(unsigned int i = 0; i < NUM_ITERS; i++) {
    if(row < BM) dest_float4[row * BN_VECTORIZED + col] = src_float4[row * N_vectorized + col];
    row += ROW_STEP;
  }
}

/// @brief Writes tile of dimension (BM x BN) from src into dst.
///        [Batches write operations through loop unrolling and instruction re-ordering]
///        [Chunks 2 doubles together to have 128b load instructions]
/// @param NUM_THREADS Number of threads per block (compile time constant)
/// @param src Pointer to src
/// @param dest Pointer to dest
/// @param N Row stride of dest
template<unsigned int BM, unsigned int BN, unsigned int NUM_THREADS, unsigned int THREAD_OFFSET>
__device__ void writeTileChunked(const unsigned int N, float* src, float* dest) {
  float4* src_float4 = reinterpret_cast<float4*>(src);
  float4* dest_float4 = reinterpret_cast<float4*>(dest);
  constexpr unsigned int BN_VECTORIZED = BN / 4;
  const unsigned int N_vectorized = N / 4;

  static_assert(NUM_THREADS % BN_VECTORIZED == 0);
  constexpr unsigned int ROW_STEP = NUM_THREADS / BN_VECTORIZED;
  // static_assert(BM % ROW_STEP == 0);
  constexpr unsigned int NUM_ITERS = BM / ROW_STEP;

  const unsigned int tId = threadIdx.y * blockDim.x + threadIdx.x - THREAD_OFFSET;
  unsigned int row = tId / BN_VECTORIZED;
  const unsigned int col = tId % BN_VECTORIZED;

  #pragma unroll
  for(unsigned int i = 0; i < NUM_ITERS; i++) {
    if(row < BM) dest_float4[row * N_vectorized + col] = src_float4[row * BN_VECTORIZED + col];
    row += ROW_STEP;
  }
}

/// @brief Loads tile of dimension (BM x BN) from src into reg(s).
///        [Batches read operations through loop unrolling and instruction re-ordering]
///        [Chunks 2 doubles together to have 128b load instructions]
///        [Loads result into registers instead of Shared Memory]
/// @param NUM_THREADS Number of threads per block (compile time constant)
/// @param src Pointer to src
/// @param dest Pointer to dest
/// @param N Row stride of src
template<unsigned int BM, unsigned int BN, unsigned int NUM_THREADS, unsigned int NUM_ITERS>
__forceinline__ __device__ void loadTileChunked(const unsigned int N, float* src, float4 regs[NUM_ITERS]) {
  float4* src_float4 = reinterpret_cast<float4*>(src);
  constexpr unsigned int BN_VECTORIZED = BN / 4;
  const unsigned int N_vectorized = N / 4;

  static_assert(NUM_THREADS % BN_VECTORIZED == 0);
  constexpr unsigned int ROW_STEP = NUM_THREADS / BN_VECTORIZED; 

  const unsigned int tId = threadIdx.y * blockDim.x + threadIdx.x;
  unsigned int row = tId / BN_VECTORIZED;
  const unsigned int col = tId % BN_VECTORIZED;

  #pragma unroll
  for(unsigned int i = 0; i < NUM_ITERS; i++) {
    regs[i] = src_float4[row * N_vectorized + col];
    row += ROW_STEP;
  }
}

/// @brief Sotres tile of dimension (BM x BN) from reg(s) into dest(s).
///        [Batches read operations through loop unrolling and instruction re-ordering]
///        [Chunks 2 doubles together to have 128b load instructions]
/// @param NUM_THREADS Number of threads per block (compile time constant)
/// @param src Pointer to src
/// @param dest Pointer to dest
/// @param N Row stride of src
template<unsigned int BM, unsigned int BN, unsigned int NUM_THREADS, unsigned int NUM_ITERS>
__forceinline__ __device__ void storeTileChunked(float4 regs[NUM_ITERS], float* dest) {
  float4* dest_float4 = reinterpret_cast<float4*>(dest);
  constexpr unsigned int BN_VECTORIZED = BN / 4;

  static_assert(NUM_THREADS % BN_VECTORIZED == 0);
  constexpr unsigned int ROW_STEP = NUM_THREADS / BN_VECTORIZED; 

  const unsigned int tId = threadIdx.y * blockDim.x + threadIdx.x;
  unsigned int row = tId / BN_VECTORIZED;
  const unsigned int col = tId % BN_VECTORIZED;

  #pragma unroll
  for(unsigned int i = 0; i < NUM_ITERS; i++) {
    dest_float4[row * BN_VECTORIZED + col] = regs[i];
    row += ROW_STEP;
  }
}

/// @brief Reads tile of dimension (BM x BN) from src into dst.
///        [Batches read operations through loop unrolling and instruction re-ordering]
///        [Chunks 2 doubles together to have 128b load instructions]
/// @param NUM_THREADS Number of threads per block (compile time constant)
/// @param src Pointer to src
/// @param dest Pointer to dest
/// @param N Row stride of src
template<unsigned int BM, unsigned int BN, unsigned int NUM_THREADS, unsigned int THREAD_OFFSET>
__forceinline__ __device__ void readTileAsync(const unsigned int N, float* src, float* dest) {
  float4* src_float4 = reinterpret_cast<float4*>(src);
  float4* dest_float4 = reinterpret_cast<float4*>(dest);
  constexpr unsigned int BN_VECTORIZED = BN / 4;
  const unsigned int N_vectorized = N / 4;

  static_assert(NUM_THREADS % BN_VECTORIZED == 0);
  constexpr unsigned int ROW_STEP = NUM_THREADS / BN_VECTORIZED; 
  // static_assert(BM % ROW_STEP == 0);
  constexpr unsigned int NUM_ITERS = (BM + ROW_STEP - 1) / ROW_STEP;

  const unsigned int tId = threadIdx.y * blockDim.x + threadIdx.x - THREAD_OFFSET;
  unsigned int row = tId / BN_VECTORIZED;
  const unsigned int col = tId % BN_VECTORIZED;

  #pragma unroll
  for(unsigned int i = 0; i < NUM_ITERS; i++) {
    if(row < BM) __pipeline_memcpy_async(&dest_float4[write_stage_idx][As_idx], src_float4[row * N_vectorized + col], sizeof(float4));
    row += ROW_STEP;
  }
}

#endif // GLOBAL_MEM_UTILS_CUH
