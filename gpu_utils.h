#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <cuda.h>
#include <stdio.h>

#ifndef GPU_DEVICE
#define GPU_DEVICE 0
#endif

void GPU_argv_init()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, GPU_DEVICE);
	cudaSetDevice(GPU_DEVICE);
	
	printf("Setting device %d: %s\n", GPU_DEVICE, prop.name);
	printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
	printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
	printf("Multiprocessors: %d\n", prop.multiProcessorCount);
	printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
	
	if (prop.major < 7) {
		printf("WARNING: This device does not support Tensor Cores (requires compute capability >= 7.0)\n");
	} else {
		printf("✓ Tensor Cores supported!\n");
		if (prop.major == 8 && prop.minor == 9) {
			printf("✓ Ada Lovelace (RTX 40-series) - 4th gen Tensor Cores!\n");
		} else if (prop.major >= 8) {
			printf("✓ Ampere or newer architecture detected\n");
		}
	}
	printf("\n");
}

#endif
