/*
 *  Copyright 2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "cuda.h"
#include "cuda_runtime.h"

// Number of samples to take on each experiment
#define NUM_SAMPLES 20

// The paramters for this experiment
#include "params_directed.h"
// Different CudaDMA implementations
#include "warp_specialized_single_buffer.h"
#include "warp_specialized_double_buffer.h"
#include "warp_specialized_manual_buffer.h"
#include "non_warp_specialized_single_buffer.h"
#include "non_warp_specialized_double_buffer.h"

// Check all CUDA calls
#define CUDA_SAFE_CALL(expr)                               \
  {                                                        \
    cudaError_t err = (expr);                              \
    if (err != cudaSuccess)                                \
    {                                                      \
      printf("CUDA error: %s\n", cudaGetErrorString(err)); \
      assert(false);                                       \
    }                                                      \
  }

// Helper method for determining alignment dependent
// on radius (assumes float2 elements)
#define PARAM_ALIGNMENT ((PARAM_RADIUS % 2) == 0 ? 16 : 8)

// Error threshold for validation
#define ERROR_THRESHOLD 1e-5f

// CPU reference implementation of the stencil computation
void stencil_2D_cpu_reference(const float2 *src_buffer, float2 *dst_buffer,
                              const int offset, const int row_stride,
                              const int slice_stride, const int z_steps)
{
  const int x_dim = PARAM_DIM_X;
  const int y_dim = PARAM_DIM_Y;
  const int radius = PARAM_RADIUS;

  for (int iz = 0; iz < z_steps; iz++)
  {
    for (int iy = 0; iy < y_dim; iy++)
    {
      for (int ix = 0; ix < x_dim; ix++)
      {
        // Calculate the index in the destination buffer
        int dst_idx = offset + iz * slice_stride + iy * row_stride + ix;

        // Perform stencil computation (same as GPU version)
#ifdef DENSE_MATH
        float2 accum;
        accum.x = 0.0f;
        accum.y = 0.0f;
        for (int dy = -radius; dy <= radius; dy++)
        {
          for (int dx = -radius; dx <= radius; dx++)
          {
            int src_idx = offset + iz * slice_stride + (iy + dy) * row_stride + (ix + dx);
            accum.x += src_buffer[src_idx].x;
            accum.y += src_buffer[src_idx].y;
          }
        }
        accum.x /= ((2 * radius + 1) * (2 * radius + 1));
        accum.y /= ((2 * radius + 1) * (2 * radius + 1));
#else
        int center_idx = offset + iz * slice_stride + iy * row_stride + ix;
        float2 accum = src_buffer[center_idx];

        // +X and -X
        for (int dx = 1; dx < radius; dx++)
        {
          int idx = offset + iz * slice_stride + iy * row_stride + (ix - dx);
          accum.x += src_buffer[idx].x;
          accum.y += src_buffer[idx].y;
        }
        for (int dx = 1; dx < radius; dx++)
        {
          int idx = offset + iz * slice_stride + iy * row_stride + (ix + dx);
          accum.x += src_buffer[idx].x;
          accum.y += src_buffer[idx].y;
        }

        // +Y and -Y
        for (int dy = 1; dy < radius; dy++)
        {
          int idx = offset + iz * slice_stride + (iy - dy) * row_stride + ix;
          accum.x += src_buffer[idx].x;
          accum.y += src_buffer[idx].y;
        }
        for (int dy = 1; dy < radius; dy++)
        {
          int idx = offset + iz * slice_stride + (iy + dy) * row_stride + ix;
          accum.x += src_buffer[idx].x;
          accum.y += src_buffer[idx].y;
        }

        accum.x /= (4 * radius + 1);
        accum.y /= (4 * radius + 1);
#endif
        dst_buffer[dst_idx] = accum;
      }
    }
  }
}

// Validate GPU output against CPU reference
int validate_output(const float2 *cpu_output, const float2 *gpu_output,
                    const int offset, const int row_stride,
                    const int slice_stride, const int z_steps)
{
  int mismatches = 0;
  const int x_dim = PARAM_DIM_X;
  const int y_dim = PARAM_DIM_Y;

  for (int iz = 0; iz < z_steps; iz++)
  {
    for (int iy = 0; iy < y_dim; iy++)
    {
      for (int ix = 0; ix < x_dim; ix++)
      {
        int idx = offset + iz * slice_stride + iy * row_stride + ix;

        float diff_x = fabs(cpu_output[idx].x - gpu_output[idx].x);
        float diff_y = fabs(cpu_output[idx].y - gpu_output[idx].y);

        if (diff_x > ERROR_THRESHOLD || diff_y > ERROR_THRESHOLD)
        {
          mismatches++;
        }
      }
    }
  }

  return mismatches;
}

enum KernelID
{
  WARP_SPECIALIZED_SINGLE_BUFFER = 0,
  WARP_SPECIALIZED_DOUBLE_BUFFER = 1,
  WARP_SPECIALIZED_MANUAL_BUFFER = 2,
  NON_WARP_SPECIALIZED_SINGLE_BUFFER = 3,
  NON_WARP_SPECIALIZED_DOUBLE_BUFFER = 4,
  NUM_KERNELS = 5,
};

int main(int argc, char **argv)
{
  CUDA_SAFE_CALL(cudaSetDevice(0));
  {
    cudaDeviceProp device_prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_prop, 0));
    fprintf(stdout, "Running CudaDMA Stencil examples on %s\n", device_prop.name);
    fprintf(stdout, "X dimension: %d\n", PARAM_DIM_X);
    fprintf(stdout, "Y dimension: %d\n", PARAM_DIM_Y);
    fprintf(stdout, "Z dimension: %d\n", PARAM_DIM_Z);
    fprintf(stdout, "TILE X: %d\n", PARAM_TILE_X);
    fprintf(stdout, "TILE Y: %d\n", PARAM_TILE_Y);
    fprintf(stdout, "RADIUS: %d\n", PARAM_RADIUS);
    fprintf(stdout, "DMA_WARPS: %d\n", PARAM_DMA_WARPS);
    fprintf(stdout, "BYTES PER THREAD: %d\n", PARAM_BYTES_PER_THREAD);
    fprintf(stdout, "\n");
  }
  // A nice trick for improving performance on Kepler if your elements
  // are at least 8 bytes
#if __CUDA_ARCH__ >= 350
  CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
#endif

  // Allocate some memory on the GPU for input and output, to keep things simple
  // we won't bother initializing it.  Pad both buffers with halo regions as if
  // so we can do have kernels go back and forth between the two in the future.
  const size_t buffer_size = (PARAM_DIM_X + 2 * PARAM_RADIUS) * (PARAM_DIM_Y + 2 * PARAM_RADIUS) * (PARAM_DIM_Z + 2 * PARAM_RADIUS) * sizeof(float2);
  fprintf(stdout, "Allocating %ld MB on the GPU\n", (2 * buffer_size / (1024 * 1024)));
  fprintf(stdout, "Allocating %ld MB on the CPU for validation\n\n", (2 * buffer_size / (1024 * 1024)));

  float2 *src_buffer_d = NULL, *dst_buffer_d = NULL;
  CUDA_SAFE_CALL(cudaMalloc((void **)&src_buffer_d, buffer_size));
  CUDA_SAFE_CALL(cudaMalloc((void **)&dst_buffer_d, buffer_size));
  assert(src_buffer_d != NULL);
  assert(dst_buffer_d != NULL);

  // Allocate CPU buffers for validation
  float2 *src_buffer_h = (float2 *)malloc(buffer_size);
  float2 *dst_buffer_cpu = (float2 *)malloc(buffer_size);
  float2 *dst_buffer_gpu = (float2 *)malloc(buffer_size);
  assert(src_buffer_h != NULL);
  assert(dst_buffer_cpu != NULL);
  assert(dst_buffer_gpu != NULL);

  // Initialize source buffer with random data
  srand(12345);
  for (size_t i = 0; i < buffer_size / sizeof(float2); i++)
  {
    src_buffer_h[i].x = (float)rand() / (float)RAND_MAX;
    src_buffer_h[i].y = (float)rand() / (float)RAND_MAX;
  }

  // Copy initialized data to GPU
  CUDA_SAFE_CALL(cudaMemcpy(src_buffer_d, src_buffer_h, buffer_size, cudaMemcpyHostToDevice));

  const int row_stride = PARAM_DIM_X + 2 * PARAM_RADIUS;
  const int slice_stride = (PARAM_DIM_Y + 2 * PARAM_RADIUS) * row_stride;
  const int offset = PARAM_RADIUS * slice_stride + PARAM_RADIUS * row_stride + PARAM_RADIUS;

  cudaStream_t timing_stream;
  CUDA_SAFE_CALL(cudaStreamCreate(&timing_stream));

  assert((PARAM_DIM_X % PARAM_TILE_X) == 0);
  assert((PARAM_DIM_Y % PARAM_TILE_Y) == 0);
  assert(((PARAM_TILE_X * PARAM_TILE_Y) % 32) == 0);

  dim3 num_ctas(PARAM_DIM_X / PARAM_TILE_X, PARAM_DIM_Y / PARAM_TILE_Y);
  // printf("Launching %dx%d threadblocks\n", num_ctas.x, num_ctas.y);

  // Array to store execution times for speedup calculation
  float kernel_times[NUM_KERNELS];

  // Run all the simulations and report the results
  for (unsigned int kernel_id = 0; kernel_id < NUM_KERNELS; kernel_id++)
  {
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start, timing_stream));
    // Run each kernel NUM_SAMPLES number of times
    switch (kernel_id)
    {
    case WARP_SPECIALIZED_SINGLE_BUFFER:
    {
      unsigned total_warps = (PARAM_TILE_X * PARAM_TILE_Y / 32) + PARAM_DMA_WARPS;
      for (unsigned int i = 0; i < NUM_SAMPLES; i++)
      {
        stencil_2D_warp_specialized_single_buffer<PARAM_ALIGNMENT, PARAM_BYTES_PER_THREAD, PARAM_TILE_X,
                                                  PARAM_TILE_Y, PARAM_RADIUS, PARAM_DMA_WARPS * 32>
            <<<num_ctas, total_warps * 32, 0, timing_stream>>>(src_buffer_d, dst_buffer_d, offset, row_stride, slice_stride, PARAM_DIM_Z);
      }
      break;
    }
    case WARP_SPECIALIZED_DOUBLE_BUFFER:
    {
      unsigned total_warps = (PARAM_TILE_X * PARAM_TILE_Y / 32) + 2 * PARAM_DMA_WARPS;
      // Number of iterations must be divisible by 2
      assert((PARAM_DIM_Z % 2) == 0);
      for (unsigned int i = 0; i < NUM_SAMPLES; i++)
      {
        stencil_2D_warp_specialized_double_buffer<PARAM_ALIGNMENT, PARAM_BYTES_PER_THREAD, PARAM_TILE_X,
                                                  PARAM_TILE_Y, PARAM_RADIUS, PARAM_DMA_WARPS * 32>
            <<<num_ctas, total_warps * 32, 0, timing_stream>>>(src_buffer_d, dst_buffer_d, offset, row_stride, slice_stride, PARAM_DIM_Z);
      }
      break;
    }
    case WARP_SPECIALIZED_MANUAL_BUFFER:
    {
      unsigned total_warps = (PARAM_TILE_X * PARAM_TILE_Y / 32) + PARAM_DMA_WARPS;
      // Number of iterations must be divisible by 2
      assert((PARAM_DIM_Z % 2) == 0);
      for (unsigned int i = 0; i < NUM_SAMPLES; i++)
      {
        stencil_2D_warp_specialized_manual_buffer<PARAM_ALIGNMENT, PARAM_BYTES_PER_THREAD, PARAM_TILE_X,
                                                  PARAM_TILE_Y, PARAM_RADIUS, PARAM_DMA_WARPS * 32>
            <<<num_ctas, total_warps * 32, 0, timing_stream>>>(src_buffer_d, dst_buffer_d, offset, row_stride, slice_stride, PARAM_DIM_Z);
      }
      break;
    }
    case NON_WARP_SPECIALIZED_SINGLE_BUFFER:
    {
      unsigned total_warps = (PARAM_TILE_X * PARAM_TILE_Y / 32);
      for (unsigned int i = 0; i < NUM_SAMPLES; i++)
      {
        stencil_2D_non_warp_specialized_single_buffer<PARAM_ALIGNMENT, PARAM_BYTES_PER_THREAD, PARAM_TILE_X,
                                                      PARAM_TILE_Y, PARAM_RADIUS>
            <<<num_ctas, total_warps * 32, 0, timing_stream>>>(src_buffer_d, dst_buffer_d, offset, row_stride, slice_stride, PARAM_DIM_Z);
      }
      break;
    }
    case NON_WARP_SPECIALIZED_DOUBLE_BUFFER:
    {
      unsigned total_warps = (PARAM_TILE_X * PARAM_TILE_Y / 32);
      // Number of iterations must be divisible by 2
      assert((PARAM_DIM_Z % 2) == 0);
      for (unsigned int i = 0; i < NUM_SAMPLES; i++)
      {
        stencil_2D_non_warp_specialized_double_buffer<PARAM_ALIGNMENT, PARAM_BYTES_PER_THREAD, PARAM_TILE_X,
                                                      PARAM_TILE_Y, PARAM_RADIUS>
            <<<num_ctas, total_warps * 32, 0, timing_stream>>>(src_buffer_d, dst_buffer_d, offset, row_stride, slice_stride, PARAM_DIM_Z);
      }
      break;
    }
    default:
      // Should never get here
      assert(false);
    }
    CUDA_SAFE_CALL(cudaEventRecord(stop, timing_stream));
    CUDA_SAFE_CALL(cudaStreamSynchronize(timing_stream));
    switch (kernel_id)
    {
    case WARP_SPECIALIZED_SINGLE_BUFFER:
      fprintf(stdout, "  RESULTS for Warp Specialized Single Buffer\n");
      break;
    case WARP_SPECIALIZED_DOUBLE_BUFFER:
      fprintf(stdout, "  RESULTS for Warp Specialized Double Buffer\n");
      break;
    case WARP_SPECIALIZED_MANUAL_BUFFER:
      fprintf(stdout, "  RESULTS for Warp Specialized Manaul Buffer\n");
      break;
    case NON_WARP_SPECIALIZED_SINGLE_BUFFER:
      fprintf(stdout, "  RESULTS for Non Warp Specialized Single Buffer\n");
      break;
    case NON_WARP_SPECIALIZED_DOUBLE_BUFFER:
      fprintf(stdout, "  RESULTS for Non Warp Specialized Double Buffer\n");
      break;
    default:
      // Should never get here
      assert(false);
    }
    float exec_time; // in milliseconds
    CUDA_SAFE_CALL(cudaEventElapsedTime(&exec_time, start, stop));
    // Scale the exec time by the number of samples
    exec_time /= NUM_SAMPLES;

    // Store execution time for speedup calculation
    kernel_times[kernel_id] = exec_time;

    fprintf(stdout, "    Execution Time: %lf ms\n", exec_time);

    unsigned total_ctas = num_ctas.x * num_ctas.y;
    double total_bytes_read = total_ctas * PARAM_DIM_Z *
                              (PARAM_TILE_Y + 2 * PARAM_RADIUS) * (PARAM_TILE_X + 2 * PARAM_RADIUS) * sizeof(float2);
    double total_bytes_written = total_ctas * PARAM_DIM_Z *
                                 (PARAM_TILE_Y * PARAM_TILE_X) * sizeof(float2);
    double load_bandwidth = (double(total_bytes_read) / double(exec_time)) * 1e-6;
    double store_bandwidth = (double(total_bytes_written) / double(exec_time)) * 1e-6;
    double full_bandwidth = (double(total_bytes_read + total_bytes_written) / double(exec_time)) * 1e-6;
    fprintf(stdout, "    READ Bandwidth:  %lf GB/s\n", load_bandwidth);
    fprintf(stdout, "    WRITE Bandwidth: %lf GB/s\n", store_bandwidth);
    fprintf(stdout, "    TOTAL Bandwidth: %lf GB/s\n", full_bandwidth);
#ifdef DENSE_MATH
    // Only print the FLOPS number if we're compute-bound
    double total_flops = total_ctas * PARAM_DIM_Z *
                         double(PARAM_TILE_Y * PARAM_TILE_X) * double((2 * PARAM_RADIUS + 1) * (2 * PARAM_RADIUS + 1) + 1);
    double flops = (double(total_flops) / double(exec_time)) * 1e-6;
    fprintf(stdout, "    Throughput: %lf GFLOPS\n\n", flops);
#endif

    // Validate GPU output against CPU reference
    // Copy GPU result back to host
    CUDA_SAFE_CALL(cudaMemcpy(dst_buffer_gpu, dst_buffer_d, buffer_size, cudaMemcpyDeviceToHost));

    // Compute CPU reference on first kernel only (all kernels should produce same result)
    if (kernel_id == 0)
    {
      fprintf(stdout, "    Computing CPU reference...\n");
      stencil_2D_cpu_reference(src_buffer_h, dst_buffer_cpu, offset, row_stride, slice_stride, PARAM_DIM_Z);
    }

    // Validate
    int mismatches = validate_output(dst_buffer_cpu, dst_buffer_gpu, offset, row_stride, slice_stride, PARAM_DIM_Z);
    fprintf(stdout, "    Validation: %d mismatches (out of %d elements)\n",
            mismatches, PARAM_DIM_X * PARAM_DIM_Y * PARAM_DIM_Z);
    if (mismatches == 0)
    {
      fprintf(stdout, "    ✓ PASSED\n\n");
    }
    else
    {
      fprintf(stdout, "    ✗ FAILED\n\n");
    }

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
  }
  CUDA_SAFE_CALL(cudaStreamDestroy(timing_stream));

  // Print speedup analysis (using NON_WARP_SPECIALIZED_SINGLE_BUFFER as baseline)
  fprintf(stdout, "\n========================================\n");
  fprintf(stdout, "SPEEDUP ANALYSIS (vs Non-Warp-Specialized Single Buffer)\n");
  fprintf(stdout, "========================================\n");
  float baseline_time = kernel_times[NON_WARP_SPECIALIZED_SINGLE_BUFFER];

  fprintf(stdout, "Warp Specialized Single Buffer: %.2fx speedup (%.3f ms vs %.3f ms)\n",
          baseline_time / kernel_times[WARP_SPECIALIZED_SINGLE_BUFFER],
          kernel_times[WARP_SPECIALIZED_SINGLE_BUFFER], baseline_time);

  fprintf(stdout, "Warp Specialized Double Buffer: %.2fx speedup (%.3f ms vs %.3f ms)\n",
          baseline_time / kernel_times[WARP_SPECIALIZED_DOUBLE_BUFFER],
          kernel_times[WARP_SPECIALIZED_DOUBLE_BUFFER], baseline_time);

  fprintf(stdout, "Warp Specialized Manual Buffer: %.2fx speedup (%.3f ms vs %.3f ms)\n",
          baseline_time / kernel_times[WARP_SPECIALIZED_MANUAL_BUFFER],
          kernel_times[WARP_SPECIALIZED_MANUAL_BUFFER], baseline_time);

  fprintf(stdout, "Non-Warp-Specialized Single Buffer: 1.00x (baseline - %.3f ms)\n",
          kernel_times[NON_WARP_SPECIALIZED_SINGLE_BUFFER]);

  fprintf(stdout, "Non-Warp-Specialized Double Buffer: %.2fx speedup (%.3f ms vs %.3f ms)\n",
          baseline_time / kernel_times[NON_WARP_SPECIALIZED_DOUBLE_BUFFER],
          kernel_times[NON_WARP_SPECIALIZED_DOUBLE_BUFFER], baseline_time);
  fprintf(stdout, "========================================\n");

  // Print out the diagnostic information for the different CudaDMA instances used

  fprintf(stdout, "\n\nPrinting diagnostic information for warp-specialized instances...\n\n");
  CudaDMAStrided<>::diagnose(PARAM_ALIGNMENT, PARAM_BYTES_PER_THREAD,
                             (PARAM_TILE_X + 2 * PARAM_RADIUS) * sizeof(float2),
                             PARAM_DMA_WARPS * 32, PARAM_TILE_Y + 2 * PARAM_RADIUS, true /*fully templated*/);

  fprintf(stdout, "\n\nPrinting diagnostic information for non-warp-specialized instances...\n\n");
  CudaDMAStrided<>::diagnose(PARAM_ALIGNMENT, PARAM_BYTES_PER_THREAD,
                             (PARAM_TILE_X + 2 * PARAM_RADIUS) * sizeof(float2),
                             (PARAM_TILE_X * PARAM_TILE_Y), PARAM_TILE_Y + 2 * PARAM_RADIUS, true /*fully templated*/);

  fprintf(stdout, "\n\nCleaning up memory and exiting...\n\n");

  // Free up our memory and finish
  CUDA_SAFE_CALL(cudaFree(src_buffer_d));
  CUDA_SAFE_CALL(cudaFree(dst_buffer_d));

  free(src_buffer_h);
  free(dst_buffer_cpu);
  free(dst_buffer_gpu);

  return 0;
}
