#ifndef Softmax_H
#define Softmax_H

#include "Activation.hpp"
#include "Tensor.hpp"
#include "Variables.hpp"

class Softmax : public Activation
{
    public:
        Softmax(std::pair<int,int> inputSize, std::string type = "softmax");
        Tensor<float> forward(Tensor<float>) override;
        Tensor<float> OMPforward(Tensor<float>) override;
        Model* copy() override;

        std::string type;
    private:
};

Model* Softmax::copy()
{
    return new Softmax(inputSize,type);
}

Softmax::Softmax(std::pair<int,int> inputSize,std::string type) : Activation(inputSize,type), type(type)
{

}

Tensor<float> Softmax::forward(Tensor<float> input)
{
    if(type=="softmax")
    {
        for(int i=0; i<input.getSize().first;i++)
        {
            float sum = 0;
            for(int j=0; j<input.getSize().second; j++)
            {
                sum += pow(Variables::e,input.data[i][j]);
            }
            for(int j=0;j<input.getSize().second;j++)
            {
                input.data[i][j] = pow(Variables::e,input.data[i][j]) / sum;
            }
        }
    }
    else if(type=="softmax2d")
    {
        float sum = 0;
        for(int i=0; i<input.getSize().first;i++)
        {
        	for(int j=0; j<input.getSize().second; j++)
        	{
    	        sum += pow(Variables::e,input.data[i][j]);
        	}
        }
        for(int i=0; i<input.getSize().first;i++)
        {
            for(int j=0;j<input.getSize().second;j++)
        	{
        	    input.data[i][j] = pow(Variables::e,input.data[i][j]) / sum;
        	}
        }    
    }
    else
    {
        throw std::runtime_error("Invalid softmax type");
    }
    return input;
}

Tensor<float> Softmax::OMPforward(Tensor<float> input)
{
    return forward(input);
}

#endif