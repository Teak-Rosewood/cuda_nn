#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void addKernel(T** a, T** b, T** c, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        c[row][col] = a[row][col] + b[row][col];
    }
}

template <typename T>
void CUDAadd(T** device_data_a, T** device_data_b, T**& device_data_added, std::pair<int, int> size) {
    // Allocate memory for the added data
    cudaMalloc(&device_data_added, size.first * sizeof(T *));
    for (int i = 0; i < size.first; ++i) {
        cudaMalloc(&device_data_added[i], size.second * sizeof(T));
    }

    // Set up the grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((size.second + blockSize.x - 1) / blockSize.x, (size.first + blockSize.y - 1) / blockSize.y);

    // Launch the addKernel
    addKernel<T><<<gridSize, blockSize>>>(device_data_a, device_data_b, device_data_added, size.first, size.second);
}