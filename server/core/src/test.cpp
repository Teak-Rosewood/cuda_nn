#include <iostream>
#include "Tensor.hpp"
#include <utility>
#include "Linear.hpp"
#include "Pipeline.hpp"
#include "Relu.hpp"
#include "MSELoss.hpp"
#include "Adam.hpp"
#include <cmath>
#include <omp.h>
#include <chrono>

using namespace std;

int main()
{
    Tensor<float> dat = Tensor<float>::readCSV("WineQT.csv");
    dat = dat.Normalize();
    std::vector<int> ind;
    ind.push_back(11);
    std::pair<Tensor<float>,Tensor<float>> vals = dat.input_output_split(ind);
    Tensor<float> input = vals.first;
    Tensor<float> output = vals.second;
    vector<Tensor<float>> input_list = input.row_split();
    vector<Tensor<float>> output_list = output.row_split();

    Pipeline myPipeline;
    Linear* q = new Linear(make_pair(1,12),6);
    Relu* r = new Relu(make_pair(1,6));
    Linear* d = new Linear(make_pair(1,6),3);
    Relu *e = new Relu(make_pair(1,3));
    Linear *f = new Linear(make_pair(1,3),1);
    Relu *g = new Relu(make_pair(1,1));

    MSELoss loss_fn;

    myPipeline.add(q);
    myPipeline.add(r);
    myPipeline.add(d);
    myPipeline.add(e);
    myPipeline.add(f);
    myPipeline.add(g);


    myPipeline.printPipeline();
    myPipeline.load("hello.arachne");
    Adam  optimizer(1e-5);

    auto start = std::chrono::high_resolution_clock::now();

    // for(int j=0;j<40;j++)
    // {
    //     float loss = 0;
    //     for(int i=0;i<input_list.size();i++)
    //     {
    //         input = input_list[i];
    //         output = output_list[i];
    //         Tensor<float> pred = myPipeline.forward(input);
    //         loss += loss_fn.loss(pred,output);
    //         myPipeline.backward(&optimizer,&loss_fn,output);
    //     }
    //     cout<<"Loss at epoch "<<j<<": "<<loss<<endl;
    // }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    int correct = 0;
    int close = 0;
    for(int i=0;i<input_list.size();i++)
    {
        input = input_list[i];
        output = output_list[i];
        Tensor<float> pred = myPipeline.forward(input);
        int actual = std::round(output.data[0][0]*5+3);
        int prediction = std::round(pred.data[0][0]*5+3);

        if(prediction == actual)
            correct++;
        if(abs(prediction-actual)<=1)
            close++;
    }

    cout<<"Accuracy: "<<float(correct)*100/input_list.size()<<"%"<<endl;
    cout<<"Close prediction: "<<float(close)*100/input_list.size()<<"%"<<endl;

    // myPipeline.save("hello");
}