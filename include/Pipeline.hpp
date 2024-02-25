#ifndef Pipeline_H
#define Pipeline_H

#include "Model.hpp"
#include <vector>
#include <iostream>
#include <utility>

class Pipeline {
public:
    Pipeline();
    void add(Model*);
    void printPipeline();

private:
    std::vector<Model*> network;
};

Pipeline::Pipeline()
{

}

void Pipeline::add(Model* model)
{
    if(network.size()>0)
    {
        Model* current = network.back();
        std::pair<int,int> current_size = current->getOutputSize();
        if (current_size != model->getInputSize())
            throw std::runtime_error("Incorrect size of weights, input size must match previous output size.");
    }
    network.push_back(model);
}

void Pipeline::printPipeline()
{
    for(Model* model: network)
    {
        std::cout<<model->type<<": "<<model->getParamCount()<<std::endl;
    }
}

#endif