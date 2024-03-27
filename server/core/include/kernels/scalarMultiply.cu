#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void scalarMultiplyKernel(T** a, T multiplicand, T** c, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        c[row][col] = a[row][col] * multiplicand;
    }
}

template <typename T>
void CUDAscalarMultiply(T** device_data, float multiplicand, T**& device_data_multiplied, std::pair<int, int> size) {
    cudaMalloc(&device_data_multiplied, size.first * sizeof(T *));
    for (int i = 0; i < size.first; ++i) {
        cudaMalloc(&device_data_multiplied[i], size.second * sizeof(T));
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((size.second + blockSize.x - 1) / blockSize.x, (size.first + blockSize.y - 1) / blockSize.y);

    scalarMultiplyKernel<T><<<gridSize, blockSize>>>(device_data, multiplicand, device_data_multiplied, size.first, size.second);
}

template void CUDAscalarMultiply<float>(float** device_data, float multiplicand, float**& device_data_multiplied, std::pair<int, int> size);