#include <stdio.h>
#include <cuda.h>

#define TILE_SIZE 32

__global__ void test_kernel() {
    printf("Block (%d,%d), Thread %d: Entering kernel\n", blockIdx.x, blockIdx.y, threadIdx.x);
}

int main() {
    dim3 block(320, 1);
    dim3 grid(1, 1);
    test_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}
