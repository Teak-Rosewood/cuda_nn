#ifndef KERNELS_H
#define KERNELS_H

template <typename T>
__global__ void transposeKernel(T** data, int width, int height);

template <typename T>
__global__ void multiplyKernel(T** a, T** b, T** c, int widthA, int heightA, int widthB);

template <typename T>
__global__ void scalarMultiplyKernel(T** a, T multiplicand, T** c, int rows, int cols);

template <typename T>
__global__ void scalarAddKernel(T** a, T to_add, T** c, int rows, int cols);

template <typename T>
__global__ void divideKernel(T** a, T** b, T** c, int rows, int cols);

template <typename T>
__global__ void addKernel(T** a, T** b, T** c, int rows, int cols);

template <typename T>
__global__ void elemMultiplyKernel(T** a, T** b, T** c, int rows, int cols);

template <typename T>
__global__ void convertFloatKernel(T** data, int rows, int cols);

template <typename T>
__global__ void sqrtKernel(T** a, float** b, int rows, int cols);

template <typename T>
__global__ void flattenKernel(T** a, T* b, int rows, int cols);

template <typename T>
__global__ void reshapeKernel(T** a, T** b, int oldCols, int newRows, int newCols);

#endif 