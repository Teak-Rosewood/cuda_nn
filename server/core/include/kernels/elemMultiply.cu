#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void elemMultiplyKernel(T** a, T** b, T** c, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        c[row][col] = a[row][col] * b[row][col];
    }
}

template <typename T>
void CUDAelemMultiply(T** device_data_a, T** device_data_b, T**& device_data_multiplied, std::pair<int, int> size) {
    // Allocate memory for the multiplied data
    cudaMalloc(&device_data_multiplied, size.first * sizeof(T *));
    for (int i = 0; i < size.first; ++i) {
        cudaMalloc(&device_data_multiplied[i], size.second * sizeof(T));
    }

    // Set up the grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((size.second + blockSize.x - 1) / blockSize.x, (size.first + blockSize.y - 1) / blockSize.y);

    // Launch the elemMultiplyKernel
    elemMultiplyKernel<T><<<gridSize, blockSize>>>(device_data_a, device_data_b, device_data_multiplied, size.first, size.second);
}
template void CUDAelemMultiply<float>(float** device_data_a, float** device_data_b, float**& device_data_multiplied, std::pair<int, int> size);