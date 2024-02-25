// Linear.hpp
#ifndef LINEAR_H
#define LINEAR_H

#include "Model.hpp"

class Linear : public Model {
public:
    Linear(int, int); // Constructor
    Tensor<float> forward() override;
    void backward() override;
};

Linear::Linear(int input_size, int output_size)
{

}

Tensor<float> Linear::forward()
{

}

void Linear::backward()
{

}

#endif
