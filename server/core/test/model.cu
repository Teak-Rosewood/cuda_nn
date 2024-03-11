#include <cstdio>
#include "model.hpp"

__global__ void initializeModel()
{
    printf("Model has been initated... \n");
}

__global__ void cudaCallFunc()
{
    printf("function has been called! \n");
}

Model::Model()
{
    initializeModel<<<1, 1>>>();
    cudaDeviceSynchronize();
};

void Model::callFunc()
{
    cudaCallFunc<<<1, 1>>>();
    cudaDeviceSynchronize();
}