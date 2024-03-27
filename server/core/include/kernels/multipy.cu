#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void multiplyKernel(T* a, T* b, T* c, int widthA, int heightA, int widthB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < heightA && col < widthB)
    {
        T sum = 0;
        for (int i = 0; i < widthA; i++)
        {
            sum += a[row * widthA + i] * b[i * widthB + col];
        }
        c[row * widthB + col] = sum;
    }
}

template <typename T>
void CUDAMultiply(T** device_data_a, T** device_data_b, T**& device_data_multiplied, std::pair<int, int> size_a, std::pair<int, int> size_b) {

    cudaMalloc(&device_data_multiplied, size_a.first * sizeof(T *));
    for (int i = 0; i < size_a.first; ++i) {
        cudaMalloc(&device_data_multiplied[i], size_b.second * sizeof(T));
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((size_b.second + blockSize.x - 1) / blockSize.x, (size_a.first + blockSize.y - 1) / blockSize.y);

    multiplyKernel<T><<<gridSize, blockSize>>>(device_data_a, device_data_b, device_data_multiplied, size_a.first, size_a.second, size_b.second);
}