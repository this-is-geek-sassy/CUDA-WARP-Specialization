#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <cuda.h>

template<unsigned int GPU_DEVICE>
size_t get_max_optin_limit()
{
	int max_optin_limit;
	cudaDeviceGetAttribute(&max_optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, GPU_DEVICE);
	return max_optin_limit;
}

template<unsigned int GPU_DEVICE>
size_t get_max_shmem_per_block()
{
	cudaSetDevice(GPU_DEVICE);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, GPU_DEVICE);
	return prop.sharedMemPerBlock;
}

template<unsigned int GPU_DEVICE>
size_t get_max_shmem_per_sm()
{
	int max_smem_per_sm;
	cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, GPU_DEVICE);
	return max_smem_per_sm;
}

template<unsigned int GPU_DEVICE>
void device_props()
{
	cudaSetDevice(GPU_DEVICE);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, GPU_DEVICE);
	
	std::cout << "Setting device " << GPU_DEVICE << ": " << prop.name << std::endl;
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
	std::cout << "Total Global Memory: " << prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 << "GB" << std::endl;
	std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
	std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << "KB" << std::endl;
	std::cout << std::endl;
}

#endif
