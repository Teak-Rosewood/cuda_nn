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
    ~Pipeline();
    void add(Model*);
    void printPipeline();
    template<typename T>
    Tensor<float> forward(Tensor<T> input);

    template<typename T>
    Tensor<float> OMPforward(Tensor<T> input);

    void backward(Optimizer*,Loss*,Tensor<float>);
    void OMPbackward(Optimizer*,Loss*,Tensor<float>);

    void save(std::string filename);
    void load(std::string filename);
private:
    std::vector<Model*> network;
    vector<Tensor<float>> graph;

    int getTrainableLayers();
};

Pipeline::Pipeline()
{

}

Pipeline::~Pipeline()
{
    for(auto model: network)
    {
        if(model!=nullptr) delete model;
    }
    network.clear();
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

void Pipeline::save(std::string filename)
{
    std::ofstream outfile(filename+".arachne");
    if (!outfile) {
        return;
    }

    for (Model* model: network) {
        if(model->trainable)
        {
            Tensor<float> tensor = *model->getWeights();
            outfile << "Tensor size: " << tensor.size.first << "x" << tensor.size.second << std::endl;
            for (int i = 0; i < tensor.size.first; ++i) {
                for (int j = 0; j < tensor.size.second; ++j) {
                    outfile << tensor.data[i][j] << " ";
                }
                outfile << std::endl;
            }
        }
    }

    outfile.close();
}

void Pipeline::load(std::string filename) 
{
    std::vector<Tensor<float>> tensors;

    std::ifstream infile(filename);
    if (!infile) {
        // Error handling: unable to open file
        throw std::runtime_error("Error opening file");
    }

    std::string line;
    // Tensor<float> tensor;
    while (std::getline(infile, line)) {
        if (line.substr(0, 11) == "Tensor size") {
            int width, height;
            sscanf(line.c_str(), "Tensor size: %dx%d", &width, &height);
            std::pair<int,int> size = std::make_pair(width, height);

            // Allocate memory for data
            float** data = new float*[width];
            for (int i = 0; i < width; ++i) {
                data[i] = new float[height];
            }

            // Read data from file
            for (int i = 0; i < width; ++i) {
                std::getline(infile, line);
                std::istringstream iss(line);
                for (int j = 0; j < height; ++j) {
                    iss >> data[i][j];
                }
            }

            tensors.push_back(Tensor<float>(data,size));

            for (int i = 0; i < width; ++i) {
                delete[] data[i];
            }
            delete[] data;
        }
    }

    infile.close();

    int j = 0;

    int count = 0;
    for(Model* model: network)
    {
        if(model->trainable)
            count++;
    }

    if(count != tensors.size())
        throw std::runtime_error("Mismatch");

    for(int i=0;i<tensors.size();i++)
    {
        for(;j<network.size();j++)
        {
            if(network[j]->trainable)
                break;
        }
        if(j>=network.size()) break;

        if(network[j]->getWeights()->getSize() == tensors[i].getSize())
        {
            network[j]->setWeights(tensors[i]);
            network[j]->getWeights()->printTensor();
        }
        else
        {
            throw std::runtime_error("Mismatch");
        }
        j++;
    }
}
#endif