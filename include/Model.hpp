#ifndef Model_H
#define Model_H

#include "Tensor.hpp" 
#include <string>

class Model {
public:
    Model(const std::string& _type = "model") : type(_type) {}
    virtual Tensor<float> forward() = 0;
    virtual void backward() = 0;
    virtual ~Model() {}
    virtual int getParamCount() = 0;
    virtual std::pair<int,int> getInputSize() = 0;
    virtual std::pair<int,int> getOutputSize() = 0;

    std::string type;
};

#endif
