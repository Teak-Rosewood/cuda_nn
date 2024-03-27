#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void divideKernel(T** a, T** b, T** c, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        c[row][col] = a[row][col] / b[row][col];
    }
}

template <typename T>
void CUDAdivide(T** device_data_a, T** device_data_b, T**& device_data_divided, std::pair<int, int> size) {
    cudaMalloc(&device_data_divided, size.first * sizeof(T *));
    for (int i = 0; i < size.first; ++i) {
        cudaMalloc(&device_data_divided[i], size.second * sizeof(T));
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((size.second + blockSize.x - 1) / blockSize.x, (size.first + blockSize.y - 1) / blockSize.y);

    divideKernel<T><<<gridSize, blockSize>>>(device_data_a, device_data_b, device_data_divided, size.first, size.second);
}
template void CUDAdivide<float>(float** device_data_a, float** device_data_b, float**& device_data_divided, std::pair<int, int> size);