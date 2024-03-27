#include <cuda_runtime.h>
#include "kernels.hpp"

template <typename T>
__global__ void convertFloatKernel(T** data, int rows, int cols)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < rows * cols)
    {
        int i = index / cols;
        int j = index % cols;
        data[i][j] = static_cast<float>(data[i][j]);
    }
}

template <typename T>
void CUDAconvertFloat(T** device_data, std::pair<int, int> size)
{
    if (std::is_same<T, float>::value)
    {
        return;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (size.first * size.second + threadsPerBlock - 1) / threadsPerBlock;

    // Call the CUDA kernel
    convertFloatKernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, size.first, size.second);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();
}
template void CUDAconvertFloat<float>(float** device_data, std::pair<int, int> size);