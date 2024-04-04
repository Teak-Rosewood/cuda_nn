#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void reshapeKernel(T** a, T** b, int oldCols, int newRows, int newCols)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < newRows * newCols)
    {
        b[index / newCols][index % newCols] = a[index / oldCols][index % oldCols];
    }
}

template <typename T>
void CUDAreshape(T** device_data, T**& device_data_reshaped, std::pair<int, int> oldSize, std::pair<int, int> newSize) {
    // Allocate memory for the reshaped data
    device_data_reshaped = new T*[newSize.first];
    for (int i = 0; i < newSize.first; ++i) {
        cudaMalloc(&device_data_reshaped[i], newSize.second * sizeof(T));
        cudaMemset(device_data_reshaped[i], 0, newSize.second * sizeof(T));
    }

    // Set up the grid and block dimensions
    int numThreads = 256;
    int numBlocks = (newSize.first * newSize.second + numThreads - 1) / numThreads;

    // Launch the reshapeKernel
    reshapeKernel<T><<<numBlocks, numThreads>>>(device_data, device_data_reshaped, oldSize.second, newSize.first, newSize.second);
    cudaDeviceSynchronize(); // Wait for CUDA to finish
}
template void CUDAreshape<float>(float** device_data, float**& device_data_reshaped, std::pair<int, int> oldSize, std::pair<int, int> newSize);