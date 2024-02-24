#include "Model.hpp"
#include <vector>

#ifndef Pipeline_H
#define Pipeline_H

class Pipeline {
public:
    template<typename... Args> void add(Args&&...);

private:
    std::vector<Model> network;
};

template<typename... Args>
void Pipeline::add(Args&&... args)
{
    (network.push_back(std::forward<Args>(args)), ...);
}

#endif