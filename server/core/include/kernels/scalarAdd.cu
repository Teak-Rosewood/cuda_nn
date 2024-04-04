#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void scalarAddKernel(T** a, T to_add, T** c, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        c[row][col] = a[row][col] + to_add;
    }
}

template <typename T>
void CUDAscalarAdd(T** device_data, T to_add, T**& device_data_added, std::pair<int, int> size) {
    device_data_added = new T*[size.first]; 
    for (int i = 0; i < size.first; ++i) {
        cudaMalloc(&device_data_added[i], size.second * sizeof(T));
        cudaMemset(device_data_added[i], 0, size.second * sizeof(T));
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((size.second + blockSize.x - 1) / blockSize.x, (size.first + blockSize.y - 1) / blockSize.y);

    scalarAddKernel<T><<<gridSize, blockSize>>>(device_data, to_add, device_data_added, size.first, size.second);
    cudaDeviceSynchronize(); // Wait for CUDA to finish
}
template void CUDAscalarAdd<float>(float** device_data, float to_add, float**& device_data_added, std::pair<int, int> size);