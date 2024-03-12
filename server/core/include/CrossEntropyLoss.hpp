#ifndef CrossEntropy_loss_H
#define CrossEntropy_loss_H

#include "Loss.hpp"
#include <cmath> // for logarithm function

class CrossEntropyLoss : public Loss {
public:
    float loss(Tensor<float> prediction, Tensor<float> actual) override;
    Tensor<float> derivative(Tensor<float> prediction, Tensor<float> actual) override;
};

float CrossEntropyLoss::loss(Tensor<float> prediction, Tensor<float> actual) {
    if (prediction.getSize() != actual.getSize())
    {
        throw std::runtime_error("Invalid dimensions");
    }
        
    float sum = 0;

    for (int i = 0; i < prediction.getSize().first; ++i) {
        for (int j = 0; j < prediction.getSize().second; ++j) {
            sum += actual.data[i][j] * log(prediction.data[i][j]) + (1 - actual.data[i][j]) * log(1 - prediction.data[i][j]);
        }
    }

    return -sum / (prediction.getSize().first * prediction.getSize().second);
}

Tensor<float> CrossEntropyLoss::derivative(Tensor<float> prediction, Tensor<float> actual) {
    Tensor<float> derivative(prediction.getSize(),0.0);

    for (int i = 0; i < prediction.getSize().first; ++i) {
        for (int j = 0; j < prediction.getSize().second; ++j) {
            float y_hat = prediction.data[i][j];
            float y = actual.data[i][j];
            derivative.data[i][j] = (y_hat - y) / (y_hat * (1 - y_hat));
        }
    }

    return derivative;
}

#endif
