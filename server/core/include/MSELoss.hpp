#ifndef Mse_loss_H
#define Mse_loss_H

#include "Loss.hpp"

class MSELoss : public Loss
{
    public:
    float loss(Tensor<float>,Tensor<float>) override;
    Tensor<float> derivative(Tensor<float>,Tensor<float>) override;
};

float MSELoss::loss(Tensor<float> prediction,Tensor<float> actual)
{
    if(prediction.getSize() != actual.getSize())
        throw std::runtime_error("Invalid dimensions");

    float sum = 0;

    for(int i=0;i<prediction.getSize().first;i++)
    {
        for(int j=0;j<prediction.getSize().second;j++)
        {
            sum += pow(prediction.data[i][j]-actual.data[i][j],2);
        }
    }

    return sum/(prediction.getSize().first * prediction.getSize().second);
}

Tensor<float> MSELoss::derivative(Tensor<float> prediction,Tensor<float> actual)
{
    return prediction - actual;
}


#endif