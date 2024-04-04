#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void transposeKernel(T** data, int width, int height)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    if (xIndex < width && yIndex < height && yIndex < xIndex)
    {
        // Swap elements at (xIndex, yIndex) and (yIndex, xIndex)
        T temp = data[xIndex][yIndex];
        data[xIndex][yIndex] = data[yIndex][xIndex];
        data[yIndex][xIndex] = temp;
    }
}

template <typename T>
void CUDATranspose(T** device_data, std::pair<int, int>& size) {

    dim3 blockSize(16, 16);
    dim3 gridSize((size.second + blockSize.x - 1) / blockSize.x, (size.first + blockSize.y - 1) / blockSize.y);

    transposeKernel<T><<<gridSize, blockSize>>>(device_data, size.second, size.first);

    size = std::make_pair(size.second, size.first);
    cudaDeviceSynchronize(); // Wait for CUDA to finish
}
template void CUDATranspose<float>(float** device_data, std::pair<int, int>& size);