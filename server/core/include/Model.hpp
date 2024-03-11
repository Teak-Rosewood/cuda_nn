#ifndef Model_H
#define Model_H

#include "Tensor.hpp" 
#include <string>

class Model {
public:
    Model(const std::string& _type = "model", const bool& trainable = false,Tensor<float>* weights = NULL) : type(_type),trainable(trainable),weights(weights) {}
    
    virtual Tensor<float> forward(Tensor<float>) = 0;
    virtual Tensor<float> OMPforward(Tensor<float>) = 0;
    void backward(Tensor<float>,Tensor<float>&,bool);
    void OMPbackward(Tensor<float>,bool);

    void computeGradients(Tensor<float>,Tensor<float>& ,bool);
    void OMPcomputeGradients(Tensor<float>,bool);

    Tensor<float> getGradients();
    virtual ~Model() {}
    virtual int getParamCount() = 0;
    virtual std::pair<int,int> getInputSize() = 0;
    virtual std::pair<int,int> getOutputSize() = 0;
    bool trainable;
    std::string type;
    bool isforward = false;
    Tensor<float>* weights;
    Tensor<float>* gradients;
    Tensor<float>* inputs;
};

void Model::backward(Tensor<float> last_gradient,Tensor<float>& gradient,bool local)
{
    if(!isforward)
        throw std::runtime_error("Forward pass must be called before backward pass");
    computeGradients(last_gradient,gradient,local);
}

void Model::OMPbackward(Tensor<float> gradient,bool local)
{
    if(!isforward)
        throw std::runtime_error("Forward pass must be called before backward pass");
    OMPcomputeGradients(gradient,local);
}

void Model::computeGradients(Tensor<float> last_gradient,Tensor<float>& gradient,bool local)
{
    Tensor<float> grad = Tensor(*inputs * last_gradient);

    if(local)
    {
        gradient.transpose();
        gradients = new Tensor(grad * gradient);
        gradient = grad;
    }
    else
    {
        gradients = new Tensor(grad);
        gradient = grad;
    }
}

void Model::OMPcomputeGradients(Tensor<float> gradient,bool local)
{
    inputs->transpose();
    gradients = new Tensor(inputs->OMPmultiply(gradient));
}

Tensor<float> Model::getGradients()
{
    return *gradients;
}

#endif
