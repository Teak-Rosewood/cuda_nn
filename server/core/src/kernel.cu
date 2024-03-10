#include <cstdio>

// CUDA kernel to print "Hello, World!" from each thread
__global__ void helloWorldKernel()
{
    printf("Hello, World! from thread %d\n", threadIdx.x);
}

void runHelloWorld()
{
    printf("Before Calling Hello World on GPU \n");
    helloWorldKernel<<<1, 20>>>();
    cudaDeviceSynchronize();
    printf("After Calling Hello World on GPU \n");
}