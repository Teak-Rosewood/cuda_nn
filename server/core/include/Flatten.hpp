#ifndef Flatten_H
#define Flatten_H

#include "Model.hpp"

class Flatten : public Model {
public:
    Flatten(std::pair<int,int> inputSize);
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

Model* Flatten::copy()
{
    return new Flatten(inputSize);
}

Flatten::Flatten(std::pair<int,int> inputSize) : Model("Flatten",false) , inputSize(inputSize) , outputSize(1,inputSize.first*inputSize.second)
{
    paramCount = 0;
}

int Flatten::getParamCount()
{
    return paramCount;
}

std::pair<int,int> Flatten::getInputSize()
{
    return inputSize;
}

std::pair<int,int> Flatten::getOutputSize()
{
    return outputSize;
}

Tensor<float> Flatten::forward(Tensor<float> input)
{
    return input.flatten();
}

Tensor<float> Flatten::OMPforward(Tensor<float> input)
{
    return forward(input);
}


#endif