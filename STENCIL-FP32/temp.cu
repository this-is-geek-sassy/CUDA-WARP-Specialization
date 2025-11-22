#include <cuda_runtime.h>

// Create texture object
cudaTextureObject_t createTexture(float *d_data, int width, int height)
{
    // Create channel descriptor
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc<float>();

    // Allocate CUDA array
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Copy data to CUDA array
    cudaMemcpy2DToArray(cuArray, 0, 0, d_data,
                        width * sizeof(float),
                        width * sizeof(float), height,
                        cudaMemcpyDeviceToDevice);

    // Specify texture resource descriptor
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear; // or cudaFilterModePoint
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0; // Use pixel coordinates

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}

// Kernel using texture
__global__ void textureKernel(cudaTextureObject_t texObj,
                              float *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // Read from texture (with hardware interpolation if linear mode)
        float value = tex2D<float>(texObj, x, y);

        output[y * width + x] = value;
    }
}

// Cleanup
void destroyTexture(cudaTextureObject_t texObj, cudaArray_t cuArray)
{
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}

int main()
{
}