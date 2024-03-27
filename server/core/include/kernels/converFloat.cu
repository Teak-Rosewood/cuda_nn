#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void convertFloatKernel(T** a, float** b, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        b[row][col] = float(a[row][col]);
    }
}

template <typename T>
void CUDAconvertFloat(T** device_data, float**& device_data_float, std::pair<int, int> size) {
    // Allocate memory for the float data
    cudaMalloc(&device_data_float, size.first * sizeof(float *));
    for (int i = 0; i < size.first; ++i) {
        cudaMalloc(&device_data_float[i], size.second * sizeof(float));
    }

    // Set up the grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((size.second + blockSize.x - 1) / blockSize.x, (size.first + blockSize.y - 1) / blockSize.y);

    // Launch the convertFloatKernel
    convertFloatKernel<T><<<gridSize, blockSize>>>(device_data, device_data_float, size.first, size.second);
}
