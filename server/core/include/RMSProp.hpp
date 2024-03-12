#ifndef RMSProp_H
#define RMSProp_H

#include "Optimizer.hpp"

class RMSProp : public Optimizer {
public:
    RMSProp(float lr = 0.001, float decay_rate = 0.9, float epsilon = 1e-7);
    void update_weights(Tensor<float>& weights, Tensor<float> gradient, int count) override;
    ~RMSProp();
private:
    float learning_rate;
    float decay_rate;
    float epsilon;
    vector<Tensor<float>*> accumulated_gradient_squared;
};

RMSProp::RMSProp(float lr, float decay_rate, float epsilon) : learning_rate(lr), decay_rate(decay_rate), epsilon(epsilon) {}

RMSProp::~RMSProp()
{
    for(auto i:accumulated_gradient_squared)
    {
        if(i != nullptr) delete i;
    }
}

void RMSProp::update_weights(Tensor<float>& weights, Tensor<float> gradient, int count) {
    if (accumulated_gradient_squared.size() < count + 1) {
        accumulated_gradient_squared.push_back(new Tensor<float>(weights.getSize(), 0.0));
    }

    // gradient.transpose();

    Tensor<float>* accum_grad_sq = accumulated_gradient_squared[count];

    *accum_grad_sq = accum_grad_sq->scalarMultiply(decay_rate) + gradient.elem_multiply(gradient).scalarMultiply(1 - decay_rate);

    weights = weights - (gradient.divide(accum_grad_sq->sqrt().scalarAdd(epsilon))).scalarMultiply(learning_rate);
}

#endif
