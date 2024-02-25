// Linear.hpp
#ifndef LINEAR_H
#define LINEAR_H

#include "Model.hpp"
#include <utility>

class Linear : public Model {
public:
    Linear(std::pair<int,int> inputSize, std::pair<int,int> outputSize);
    int getParamCount() override;
    std::pair<int,int> getInputSize() override;
    std::pair<int,int> getOutputSize() override;
    Tensor<float> forward() override;

    void backward() override;
private:
    std::pair<int,int> inputSize;
    std::pair<int,int> outputSize;
    int paramCount;
};

Linear::Linear(std::pair<int,int> inputSize, std::pair<int,int> outputSize) : Model("Linear"),inputSize(inputSize), outputSize(outputSize)
{
    paramCount = inputSize.second * outputSize.first;
}

int Linear::getParamCount()
{
    return paramCount;
}

Tensor<float> Linear::forward()
{

}

void Linear::backward()
{
}

std::pair<int,int> Linear::getOutputSize()
{
    return outputSize;
}

std::pair<int,int> Linear::getInputSize()
{
    return inputSize;
}

#endif
