#ifndef Model_H
#define Model_H

#include "Tensor.hpp" 
class Model {
public:
    virtual Tensor<float> forward() = 0;
    virtual void backward() = 0;
    virtual ~Model() {}
};

#endif
