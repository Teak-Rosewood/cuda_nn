#ifndef Loss_H
#define Loss_H

#include "Tensor.hpp"

class Loss
{
    public:
        virtual float loss(Tensor<float>,Tensor<float>) = 0;
        virtual Tensor<float> derivative(Tensor<float>,Tensor<float>) = 0;
};

#endif