#ifndef Pipeline_H
#define Pipeline_H

#include "Model.hpp"
#include <vector>
#include <iostream>
#include <utility>
#include <unordered_map>
#include "Optimizer.hpp"
#include "Loss.hpp"
#include "Activation.hpp"

class Pipeline {
public:
    Pipeline();
    void add(Model*);
    void printPipeline();
    template<typename T>
    Tensor<float> forward(Tensor<T> input);

    template<typename T>
    Tensor<float> OMPforward(Tensor<T> input);

    void backward(Optimizer*,Loss*,Tensor<float>);
    void OMPbackward(Optimizer*,Loss*,Tensor<float>);
private:
    std::vector<Model*> network;
    vector<Tensor<float>> graph;

    int getTrainableLayers();
};

Pipeline::Pipeline()
{

}

int Pipeline::getTrainableLayers()
{
    int count = 0;
    for(Model* model:network)
    {
        if(model->trainable)
            count++;
    }
    return count;
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

    if(network.size()>0 && network.back()->trainable)
    {
        if (dynamic_cast<Activation*>(model))
            network.push_back(model);
        else
            throw std::runtime_error("Trainable layers must be followed by an activation function.");
    }
    else if(network.size()>0 && !network.back()->trainable && dynamic_cast<Activation*>(model))
    {
        throw std::runtime_error("Activation layer requires trainable layer preceeding it.");
    }
    else if(getTrainableLayers()==0 && dynamic_cast<Activation*>(model))
    {
        throw std::runtime_error("Activation layer requires trainable layer preceeding it.");
    }
    else
        network.push_back(model);
}

void Pipeline::printPipeline()
{
    int total_parameter_count = 0;
    std::unordered_map<std::string,int> count_check;
    std::cout<<"Layer"<<"\t\t"<<"Input"<<"\t\t"<<"Output"<<"\t\t"<<"Parameter Count"<<std::endl;
    for(Model* model: network)
    {
        total_parameter_count += model->getParamCount();
        count_check[model->type]+=1;
        std::cout<<model->type<<" "<<count_check[model->type]<<":"<<"\t\t"<<"("<<model->getInputSize().first<<","<<model->getInputSize().second<<")"<<"\t\t"<<"("<<model->getOutputSize().first<<","<<model->getOutputSize().second<<")"<<"\t\t"<<model->getParamCount()<<std::endl;
    }
    std::cout<<"Total Parameter Count:"<<"\t"<<total_parameter_count<<std::endl;
}

template<typename T>
Tensor<float> Pipeline::forward(Tensor<T> input)
{
    if(network.back()->trainable)
        throw std::runtime_error("Last layer must be an activation layer, use Linear activation to maintain outputs");
    Tensor<float> matrix = input.convertFloat();
    graph.push_back(matrix);
    for(Model* model: network)
    {
        matrix = model->forward(matrix);
        if (dynamic_cast<Activation*>(model))
            graph.push_back(matrix);
    }
    return matrix;
}

template<typename T>
Tensor<float> Pipeline::OMPforward(Tensor<T> input)
{
    if(network.back()->trainable)
        throw std::runtime_error("Last layer must be an activation layer, use Linear activation to maintain outputs");
    Tensor<float> matrix = input.convertFloat();
    graph.push_back(matrix);
    for(Model* model: network)
    {
        matrix = model->OMPforward(matrix);
        if (dynamic_cast<Activation*>(model))
            graph.push_back(matrix);
    }
    return matrix;
}

void Pipeline::backward(Optimizer* optimizer, Loss* loss, Tensor<float> actual)
{
    Tensor<float> last_gradient = loss->derivative(graph.back(),actual);\
    Tensor<float> gradient = Tensor<float>(make_pair(1,1),0.0);
    int start = 0;
    for (int i = network.size() - 1; i >= 0; --i) 
    {
        if(network[i]->trainable)
        {
            if(start==0)
            {
                network[i]->backward(last_gradient,gradient,false);
                start++;
            }
            else
            {
                network[i]->backward(last_gradient,gradient,true);
            }
        }
    }

    int count = 0;
    for (int i = network.size() - 1; i >= 0; --i) 
    {
        if(network[i]->trainable)
        {
            optimizer->update_weights(*network[i]->weights,*network[i]->gradients,count);
            count++;
        }
            
    }
}

void Pipeline::OMPbackward(Optimizer* optimizer, Loss* loss, Tensor<float> actual)
{
    Tensor<float> gradient = loss->derivative(graph.back(),actual);
    for (int i = network.size() - 1; i >= 0; --i) 
    {
        if(network[i]->trainable)
            network[i]->OMPbackward(gradient,false);
    }

    int count = 0;
    for (int i = network.size() - 1; i >= 0; --i) 
    {
        if(network[i]->trainable)
        {
            optimizer->update_weights(*network[i]->weights,*network[i]->gradients,count);
            count++;
        }
            
    }
}

#endif