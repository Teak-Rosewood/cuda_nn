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
    ~Model();

    Tensor<float>* getWeights();
    void setWeights(Tensor<float>);

    virtual Model* copy() = 0;

    virtual int getParamCount() = 0;
    virtual std::pair<int,int> getInputSize() = 0;
    virtual std::pair<int,int> getOutputSize() = 0;
    bool trainable;
    std::string type;
    bool isforward = false;
    Tensor<float>* weights = nullptr;
    Tensor<float>* gradients = nullptr;
    Tensor<float>* inputs = nullptr;
};


Model::~Model() {
    if (trainable) {
        if (weights != nullptr) delete weights;
        if (gradients != nullptr) delete gradients;
        if (inputs != nullptr) delete inputs;
    }
}

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
    Tensor<float> grad = Tensor <float> (*inputs * last_gradient);

    if(local)
    {
        gradient.transpose();
        if (gradients != nullptr) delete gradients;
        gradients = new Tensor <float> (grad * gradient);
        gradient = grad;
    }
    else
    {
        if (gradients != nullptr) delete gradients;
        gradients = new Tensor <float> (grad);
        gradient = grad;
    }
}

void Model::OMPcomputeGradients(Tensor<float> gradient,bool local)
{
    inputs->transpose();
    gradients = new Tensor <float> (inputs->OMPmultiply(gradient));
}

Tensor<float> Model::getGradients()
{
    return *gradients;
}

Tensor<float>* Model::getWeights()
{
    return weights;
}

void Model::setWeights(Tensor<float> new_weights)
{
    if(weights!=nullptr) delete weights;
    weights = new Tensor<float>(new_weights);
}

#endif
