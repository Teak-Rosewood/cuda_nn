#include "Model.hpp"
#include "Tensor.hpp"

#ifndef Linear_H
#define Linear_H

class Linear : public Model
{
    public:
        Tensor<float> forward() override;
}

Tensor<float> Linear::forward() override
{

}

#endif