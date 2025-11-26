#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <cuda.h>

template<unsigned int GPU_DEVICE>
void device_props()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, GPU_DEVICE);
	cudaSetDevice(GPU_DEVICE);
	
	std::cout << "Setting device " << GPU_DEVICE << ": " << prop.name << std::endl;
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
	std::cout << "Total Global Memory: " << prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 << std::endl;
	std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
	std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << std::endl;
	std::cout << std::endl;
}

#endif
