#ifndef Activation_H
#define Activation_H

#include "Model.hpp"
#include "Tensor.hpp"
#include <unordered_map>


class Activation : public Model {
    public:
        Activation(std::pair<int,int>, const std::string& _type = "activation");
        virtual Tensor<float> forward(Tensor<float>) = 0;
        virtual Tensor<float> OMPforward(Tensor<float>) = 0;
        int getParamCount() override;
        std::pair<int,int> getInputSize() override;
        std::pair<int,int> getOutputSize() override;
        virtual Model* copy() = 0;

        std::pair<int,int> inputSize;
    private:
};

Activation::Activation(std::pair<int,int> inputSize, const std::string& _type) : Model(_type,false), inputSize(inputSize) {

}

int Activation::getParamCount()
{
    return 0;
}

std::pair<int,int> Activation::getInputSize()
{
    return inputSize;
}

std::pair<int,int> Activation::getOutputSize()
{
    return inputSize;
}

#endif
