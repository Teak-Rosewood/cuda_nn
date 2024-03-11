#ifndef Activation_H
#define Activation_H

#include "Model.hpp"
#include "Tensor.hpp"
#include <unordered_map>


class Activation : public Model {
    public:
        Activation(std::pair<int,int>, const std::string& _type = "activation");
        int getParamCount() override;
        std::pair<int,int> getInputSize() override;
        std::pair<int,int> getOutputSize() override;

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


// // Calls a map to the relu helper function
// void Activation::Relu(Tensor<float>& input)
// {
//     input.map(Activation::relu);
// }

// // InPlace edits the values using f(x) = { x if x>0
// //                                       { 0 else
// void Activation::relu(float* input)
// {
//     if(*input<0)
//         *input = 0;
// }

// // No influence on tensor f(x) = x
// void Activation::Linear(Tensor<float>& input)
// {
    
// }

// void Activation::Softmax(Tensor<float>& input)
// {
//     for(int i=0; i<input.getSize().first;i++)
//     {
//     	float sum = 0;
//     	for(int j=0; j<input.getSize().second; j++)
//     	{
// 	    sum += pow(Variables::e,input.data[i][j]);
//     	}
//     	for(int j=0;j<input.getSize().second;j++)
//     	{
//     	    input.data[i][j] = pow(Variables::e,input.data[i][j]) / sum;
//     	}
//     }
// }

// void Activation::Softmax2d(Tensor<float>& input)
// {
//     float sum = 0;
//     for(int i=0; i<input.getSize().first;i++)
//     {
//     	for(int j=0; j<input.getSize().second; j++)
//     	{
// 	    sum += pow(Variables::e,input.data[i][j]);
//     	}
//     }
//     for(int i=0; i<input.getSize().first;i++)
//     {
//         for(int j=0;j<input.getSize().second;j++)
//     	{
//     	    input.data[i][j] = pow(Variables::e,input.data[i][j]) / sum;
//     	}
//     }
// }

// void Activation::LeakyRelu(Tensor<float>& input)
// {
//     input.map(Activation::leakyrelu);
// }

// void Activation::leakyrelu(float* input)
// {
//     if(*input<0)
//         *input = -0.1 * *input;
// }
#endif
