// Linear.hpp
#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "Model.hpp"
#include <utility>
#include <cmath>

class Normalize : public Model {
public:
    Normalize(std::pair<int,int> inputSize);
    int getParamCount() override;
    std::pair<int,int> getInputSize() override;
    std::pair<int,int> getOutputSize() override;
    Tensor<float> forward(Tensor<float>) override;
    Tensor<float> OMPforward(Tensor<float>) override;
    Model* copy() override;

private:
    std::pair<int,int> inputSize;
    std::pair<int,int> outputSize;
    int paramCount;
};

Model* Normalize::copy()
{
    return new Normalize(inputSize);
}

Normalize::Normalize(std::pair<int,int> inputSize) : Model("Normalization",false), inputSize(inputSize)
{
    paramCount = 0;
    this->inputSize = make_pair(inputSize.first,inputSize.second);
    this->outputSize = make_pair(inputSize.first,inputSize.second);
}

int Normalize::getParamCount()
{
    return paramCount;
}

Tensor<float> Normalize::forward(Tensor<float> input)
{
    float max = input.max();
    float min = input.min();
    Tensor<float> newCopy = input.copy();
    for(int i=0;i<newCopy.getSize().first;i++)
    {
        for(int j=0;j<newCopy.getSize().second;j++)
        {
             newCopy.data[i][j] = (newCopy.data[i][j] - min)/(max-min);
        }
    }
    
    return newCopy;
}

Tensor<float> Normalize::OMPforward(Tensor<float> input)
{
    return forward(input);
}


std::pair<int,int> Normalize::getOutputSize()
{
    return outputSize;
}

std::pair<int,int> Normalize::getInputSize()
{
    return inputSize;
}


#endif
