#include "Tensor.hpp"

#ifndef Model_H
#define Model_H

class Model
{
    public:
        virtual Tensor<float> forward();
        virtual void backward();
};

#endif