#include "Tensor.hpp"
#include "Activation.hpp"

#ifndef Relu_H
#define Relu_H

class Relu : public Activation
{
    public:
    void activate() override;
};

#endif