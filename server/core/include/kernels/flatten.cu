#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void flattenKernel(T** a, T* b, int rows, int cols)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < rows * cols)
    {
        b[index] = a[index / cols][index % cols];
    }
}

template <typename T>
void CUDAflatten(T** device_data, T*& device_data_flattened, std::pair<int, int> size) {
    // Allocate memory for the flattened data
    cudaMalloc(&device_data_flattened, size.first * size.second * sizeof(T));

    // Set up the grid and block dimensions
    int numThreads = 256;
    int numBlocks = (size.first * size.second + numThreads - 1) / numThreads;

    // Launch the flattenKernel
    flattenKernel<T><<<numBlocks, numThreads>>>(device_data, device_data_flattened, size.first, size.second);
}
template void CUDAflatten<float>(float** device_data, float*& device_data_flattened, std::pair<int, int> size);