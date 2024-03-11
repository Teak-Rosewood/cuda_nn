#ifndef SGD_H
#define SGD_H

#include "Optimizer.hpp"

class SGD : public Optimizer
{
    public:
    SGD(float=1e-2,float=0);
    void update_weights(Tensor<float>&,Tensor<float>,int) override;

    private:
    float learning_rate;
    float weight_decay;
};

SGD::SGD(float alpha, float weight_decay) : learning_rate(alpha),weight_decay(weight_decay) {}


void SGD::update_weights(Tensor<float>& weights, Tensor<float> gradient,int count)
{
    // gradient.transpose();
    weights = weights - (gradient+weights.scalarMultiply(weight_decay)).scalarMultiply(learning_rate);
}

#endif