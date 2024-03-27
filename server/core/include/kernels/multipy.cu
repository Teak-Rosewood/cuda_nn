#include <cuda_runtime.h>
#include "kernels.hpp"
template <typename T>
__global__ void multiplyKernel(T** a, T** b, T** c, int widthA, int heightA, int widthB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < heightA && col < widthB)
    {
        T sum = 0;
        for (int i = 0; i < widthA; i++)
        {
            sum += a[row][i] * b[i][col];
        }
        c[row][col] = sum;
    }
}

template <typename T>
void CUDAMultiply(T** device_data_a, T** device_data_b, T**& device_data_multiplied, std::pair<int, int> size_a, std::pair<int, int> size_b) {
    // Define block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((size_b.second + blockSize.x - 1) / blockSize.x, (size_a.first + blockSize.y - 1) / blockSize.y);

    // Allocate memory for the output matrix on the device
    device_data_multiplied = new T*[size_a.first];
    for (int i = 0; i < size_a.first; ++i) {
        cudaMalloc(&device_data_multiplied[i], size_b.second * sizeof(T));
        cudaMemset(device_data_multiplied[i], 0, size_b.second * sizeof(T));
    }

    // Call the kernel function
    multiplyKernel<T><<<gridSize, blockSize>>>(device_data_a, device_data_b, device_data_multiplied, size_a.first, size_a.second, size_b.second);

    // Free the host pointers
    cudaDeviceSynchronize();
}
template void CUDAMultiply<float>(float** device_data_a, float** device_data_b, float**& device_data_multiplied, std::pair<int, int> size_a, std::pair<int, int> size_b);