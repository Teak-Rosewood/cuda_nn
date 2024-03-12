// Linear.hpp
#ifndef LINEAR_H
#define LINEAR_H

#include "Model.hpp"
#include <utility>
#include <cmath>

class Linear : public Model {
public:
    Linear(std::pair<int,int> inputSize, int outputSize);
    int getParamCount() override;
    std::pair<int,int> getInputSize() override;
    std::pair<int,int> getOutputSize() override;
    Tensor<float> forward(Tensor<float>) override;
    Tensor<float> OMPforward(Tensor<float>) override;
    Model* copy() override;

    void printWeights();

private:
    std::pair<int,int> inputSize;
    std::pair<int,int> outputSize;
    int paramCount;
    std::pair<int,int> weight_size;
};

Model* Linear::copy(){
    return new Linear(inputSize,outputSize.second);
}

Linear::Linear(std::pair<int,int> inputSize, int outputSize) : Model("Linear",true),inputSize(inputSize), outputSize(make_pair(inputSize.first,outputSize))
{
    weight_size = make_pair(inputSize.second,outputSize);
    paramCount = weight_size.second * weight_size.first;
    float value = std::sqrt(6.0/((inputSize.first*inputSize.second)+(outputSize*inputSize.second)));
    float **data = new float*[weight_size.first];
    for(int i=0;i<weight_size.first;i++)
    {
        data[i] = new float[weight_size.second];
    }

    for(int i=0;i<weight_size.first;i++)
    {
        for(int j=0;j<weight_size.second;j++)
        {
            data[i][j] = value;
        }
    }

    weights = new Tensor<float>(data,weight_size);

    for(int i=0;i<weight_size.first;i++)
    {
        delete[] data[i];
    }

    delete[] data;
}

int Linear::getParamCount()
{
    return paramCount;
}

Tensor<float> Linear::forward(Tensor<float> input)
{
    // *inputs = input.copy();
    Tensor<float> next_val = input*(*weights);
    if (inputs != nullptr) delete inputs;
    inputs = new Tensor <float> (input);
    inputs->transpose();
    isforward = true;
    return next_val;
}

Tensor<float> Linear::OMPforward(Tensor<float> input)
{
    // *inputs = input.copy();
    Tensor<float> next_val = input.OMPmultiply(*weights);
    inputs = new Tensor<float>(input);
    inputs->transpose();
    isforward = true;
    return next_val;
}


std::pair<int,int> Linear::getOutputSize()
{
    return outputSize;
}

std::pair<int,int> Linear::getInputSize()
{
    return inputSize;
}

void Linear::printWeights()
{
    for(int i=0;i<weight_size.first;i++)
    {
        for(int j=0;j<weight_size.second;j++)
        {
            std::cout<<this->weights->data[i][j];
            if(j!=weight_size.second-1)
                std::cout<<",";
        }
        std::cout<<std::endl;
    }
}

#endif
