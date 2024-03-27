#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void transposeKernel(T* input, T* output, int width, int height)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIndex < width && yIndex < height)
    {
        int pos = yIndex * width + xIndex;
        int trans_pos = xIndex * height + yIndex;
        output[trans_pos] = input[pos];
    }
}

template <typename T>
void CUDATranspose(T** device_data, T**& device_data_transposed, std::pair<int, int>& size) {

    cudaMalloc(&device_data_transposed, size.first * size.second * sizeof(T));

    dim3 blockSize(16, 16);
    dim3 gridSize((size.second + blockSize.x - 1) / blockSize.x, (size.first + blockSize.y - 1) / blockSize.y);

    transposeKernel<T><<<gridSize, blockSize>>>(device_data, device_data_transposed, size.second, size.first);

    for (int i = 0; i < size.first; i++) {
        cudaFree(device_data[i]);
    }
    cudaFree(device_data);

    size = std::make_pair(size.second, size.first);
}