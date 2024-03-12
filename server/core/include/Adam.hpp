#ifndef Adam_H
#define Adam_H

#include "Optimizer.hpp"


//Needs fixing
class Adam : public Optimizer {
public:
    Adam(float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-7);
    ~Adam();
    void update_weights(Tensor<float>&, Tensor<float>,int) override;

private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    vector<Tensor<float>*> m_v; 
    vector<Tensor<float>*> v_v;
    vector<int> t;
};

Adam::Adam(float lr, float beta1, float beta2, float epsilon) : learning_rate(lr), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

Adam::~Adam()
{
    for(auto i:m_v)
    {
        if(i != nullptr) delete i;
    }

    for(auto i:v_v)
    {
        if(i != nullptr) delete i;
    }
}

void Adam::update_weights(Tensor<float>& weights, Tensor<float> gradient, int count) {
    if(t.size() < count+1)
    {
        t.push_back(0);
        m_v.push_back(new Tensor<float>(weights.getSize(),0.0));
        v_v.push_back(new Tensor<float>(weights.getSize(),0.0));
    }

    // gradient.transpose();

    Tensor<float>* m = m_v[count];
    Tensor<float>* v = v_v[count];

    t[count]++;

    *m = m->scalarMultiply(beta1) + gradient.scalarMultiply(1-beta1);
    *v = v->scalarMultiply(beta2) + gradient.elem_multiply(gradient).scalarMultiply(1-beta2);

    Tensor<float> m_hat = m->scalarMultiply(1.0/(1.0 - pow(beta1, t[count])));
    Tensor<float> v_hat = v->scalarMultiply(1.0/(1.0 - pow(beta2, t[count])));

    weights = weights - (m_hat.divide(v_hat.sqrt().scalarAdd(epsilon))).scalarMultiply(learning_rate);
}



#endif