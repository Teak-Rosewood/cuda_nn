#ifndef Tensor_H
#define Tensor_H

#include <utility>
#include "Tensor.hpp"
#include <utility>
#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include "cuda_runtime.h"
#include "kernels/kernels.hpp"

using namespace std;

template <typename T>
class Tensor
{
public:
    // Constructor
    Tensor(T **, pair<int, int>);
    Tensor(const Tensor &other);
    Tensor(pair<int, int>, T);
    Tensor() : data(nullptr), size(0, 0) {}
    static Tensor<T> readCSV(const std::string &filename);

    Tensor<T> Normalize();
    // Matrix operations
    Tensor<T> multiply(Tensor<T>);
    Tensor<T> OMPmultiply(Tensor<T>);

    Tensor<T> elem_multiply(Tensor<T>);

    Tensor<T> scalarMultiply(float);
    Tensor<T> OMPscalarMultiply(float);

    Tensor<T> add(Tensor<T>);
    Tensor<T> OMPadd(Tensor<T>);

    Tensor<T> scalarAdd(T);

    Tensor<T> divide(Tensor<T>);

    Tensor<float> convertFloat();
    Tensor<T> flatten();
    Tensor<T> reshape(pair<int, int>);

    Tensor<float> sqrt();

    static Tensor<int> randomTensor(std::pair<int, int>);
    static Tensor<int> randomTensor(std::pair<int, int>, int min, int max);

    static Tensor<float> randomFloatTensor(std::pair<int, int>);

    // CUDA Memory Management
    void moveToDevice();
    void moveToHost();

    vector<Tensor<T>> row_split();
    std::pair<Tensor<T>, Tensor<T>> input_output_split(vector<int>);

    ~Tensor();

    T max();
    std::pair<int, int> argmax();
    T min();
    std::pair<int, int> argmin();

    // Matrix tranformations
    void transpose();
    void OMPtranspose();

    // void inverse();
    // void OMPinverse();

    void map(void (*func)(T *));

    // Copy function
    Tensor<T> copy();

    // Operations
    Tensor &operator=(const Tensor &other);
    Tensor operator+(Tensor other);
    Tensor operator-(Tensor other);
    Tensor operator*(Tensor other);

    // Helper functions
    pair<int, int> getSize();
    void printSize();
    void printTensor();
    T **data;
    T **device_data = nullptr;
    int device_count = 0;
    pair<int, int> size;

private:
    // static void swap(T*,T*);
    static bool find(vector<int>, int);
};

template <typename T>
Tensor<T>::~Tensor()
{

    if (device_data != nullptr)
    {
        this->moveToHost();
    }
    if (data != nullptr)
    {
        for (int i = 0; i < this->size.first; ++i)
        {
            if (data[i] != nullptr)
                delete[] data[i];
        }
        delete[] data;
    }
}

template <typename T>
Tensor<T>::Tensor(T **data, pair<int, int> size)
{
    // Initialize data and size here if needed
    this->size = size;
    this->data = new T *[size.first];
    for (int i = 0; i < size.first; ++i)
    {
        this->data[i] = new T[size.second];
    }

    for (int i = 0; i < size.first; i++)
    {
        for (int j = 0; j < size.second; j++)
        {
            this->data[i][j] = *(*(data + i) + j);
        }
    }

    cudaGetDeviceCount(&this->device_count);
    if (this->device_count > 0)
    {
        this->moveToDevice();
    }
}

template <typename T>
Tensor<T>::Tensor(pair<int, int> size, T data)
{
    // Initialize data and size here if needed
    this->size = size;
    this->data = new T *[size.first];
    for (int i = 0; i < size.first; ++i)
    {
        this->data[i] = new T[size.second];
        for (int j = 0; j < size.second; j++)
        {
            this->data[i][j] = data;
        }
    }
    cudaGetDeviceCount(&this->device_count);
    if (this->device_count > 0)
    {
        this->moveToDevice();
    }
}

template <typename T>
void Tensor<T>::printSize()
{
    cout << "(" << size.first << "," << size.second << ")";
}

template <typename T>
pair<int, int> Tensor<T>::getSize()
{
    return this->size;
}

template <typename T>
void Tensor<T>::printTensor()
{
    if (this->device_count > 0)
    {
        this->moveToHost();
    }

    for (int i = 0; i < size.first; i++)
    {
        for (int j = 0; j < size.second; j++)
        {
            cout << this->data[i][j];
            if (j != size.second - 1)
                cout << ",";
        }
        cout << endl;
    }
}

template <typename T>
void Tensor<T>::OMPtranspose()
{
    this->size = make_pair(this->size.second, this->size.first);
    T **temp_data = new T *[this->size.first];

    for (int i = 0; i < this->size.first; ++i)
    {
        temp_data[i] = new T[this->size.second];
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < this->size.second; i++)
    {
        for (int j = 0; j < this->size.first; j++)
        {
            temp_data[j][i] = this->data[i][j];
        }
    }

    T **to_delete = this->data;
    for (int i = 0; i < this->size.second; i++)
    {
        delete[] to_delete[i];
    }
    delete[] to_delete;

    this->data = temp_data;
}

template <typename T>
void Tensor<T>::transpose()
{
    if (this->device_count > 0)
    {
        T **device_data_transposed;
        CUDATranspose<T>(this->device_data, device_data_transposed, this->size);
        this->device_data = device_data_transposed;
    }
    else
    {
        this->size = make_pair(this->size.second, this->size.first);
        T **temp_data = new T *[this->size.first];

        for (int i = 0; i < this->size.first; ++i)
        {
            temp_data[i] = new T[this->size.second];
        }

        for (int i = 0; i < this->size.second; i++)
        {
            for (int j = 0; j < this->size.first; j++)
            {
                temp_data[j][i] = this->data[i][j];
            }
        }

        T **to_delete = this->data;
        for (int i = 0; i < this->size.second; i++)
        {
            delete[] to_delete[i];
        }
        delete[] to_delete;

        this->data = temp_data;
    }
}

template <typename T>
Tensor<T> Tensor<T>::multiply(Tensor<T> b)
{
    if (this->size.second != b.size.first)
    {
        this->printSize();
        b.printSize();
        throw std::invalid_argument("Incorrect size for matrix multiplication, must be of type - (" + to_string(this->getSize().first) + "," + to_string(this->getSize().second) + ")x(" + to_string(b.getSize().first) + "," + to_string(b.getSize().second) + ")");
    }

    T **multiplied;
    pair<int, int> ml_size = make_pair(this->size.first, b.size.second);

    if (this->device_count > 0)
    {
        T **device_data_multiplied;
        CUDAMultiply<T>(this->device_data, b.device_data, device_data_multiplied, this->size, b.size);

        Tensor<T> output(ml_size, 0);
        output.device_data = device_data_multiplied;
        output.moveToHost();

        return output;
    }
    else
    {
        multiplied = new T *[this->size.first];
        for (int i = 0; i < this->size.first; i++)
        {
            multiplied[i] = new T[b.size.second];
            for (int j = 0; j < b.size.second; j++)
            {
                multiplied[i][j] = 0;
            }
        }

        for (int i = 0; i < this->size.first; i++)
        {
            for (int j = 0; j < b.size.second; j++)
            {
                for (int k = 0; k < b.size.first; k++)
                {
                    multiplied[i][j] += this->data[i][k] * b.data[k][j];
                }
            }
        }

        Tensor<T> output(multiplied, ml_size);

        for (int i = 0; i < this->size.first; i++)
        {
            delete[] multiplied[i];
        }
        delete[] multiplied;

        return output;
    }
}

template <typename T>
Tensor<T> Tensor<T>::OMPmultiply(Tensor<T> b)
{
    if (this->size.second != b.size.first)
        throw std::invalid_argument("Incorrect size for matrix multiplication, must be of type - (a,b)x(b,c)");

    T **multiplied = new T *[this->size.first];
    pair<int, int> ml_size = make_pair(this->size.first, b.size.second);

    for (int i = 0; i < this->size.first; i++)
    {
        multiplied[i] = new T[b.size.second];
        for (int j = 0; j < b.size.second; j++)
        {
            multiplied[i][j] = 0;
        }
    }

#pragma omp parallel for collapse(3)
    for (int i = 0; i < this->size.first; i++)
    {
        for (int j = 0; j < b.size.second; j++)
        {
            for (int k = 0; k < b.size.first; k++)
            {
                multiplied[i][j] += this->data[i][k] * b.data[k][j];
            }
        }
    }

    Tensor<T> output(multiplied, ml_size);

    for (int i = 0; i < this->size.first; i++)
    {
        delete[] multiplied[i];
    }
    delete[] multiplied;

    return output;
}

template <typename T>
Tensor<T> Tensor<T>::scalarMultiply(float multiplicand)
{
    T **multiplied;
    pair<int, int> ml_size = make_pair(this->size.first, this->size.second);

    if (this->device_count > 0)
    {
        T **device_data_multiplied;
        CUDAscalarMultiply<T>(this->device_data, multiplicand, device_data_multiplied, this->size);

        Tensor<T> output(ml_size, 0);
        output.device_data = device_data_multiplied;

        return output;
    }
    else
    {
        multiplied = new T *[this->size.first];
        for (int i = 0; i < this->size.first; i++)
        {
            multiplied[i] = new T[this->size.second];
            for (int j = 0; j < this->size.second; j++)
            {
                multiplied[i][j] = this->data[i][j] * multiplicand;
            }
        }

        Tensor<T> output(multiplied, ml_size);

        for (int i = 0; i < this->size.first; i++)
        {
            delete[] multiplied[i];
        }
        delete[] multiplied;

        return output;
    }
}

template <typename T>
Tensor<T> Tensor<T>::scalarAdd(T to_add)
{
    T **added;
    pair<int, int> ml_size = make_pair(this->size.first, this->size.second);

    if (this->device_count > 0)
    {
        T **device_data_added;
        CUDAscalarAdd<T>(this->device_data, to_add, device_data_added, this->size);

        Tensor<T> output(ml_size, 0);
        output.device_data = device_data_added;

        return output;
    }
    else
    {
        added = new T *[this->size.first];
        for (int i = 0; i < this->size.first; i++)
        {
            added[i] = new T[this->size.second];
            for (int j = 0; j < this->size.second; j++)
            {
                added[i][j] = this->data[i][j] + to_add;
            }
        }

        Tensor<T> output(added, ml_size);

        for (int i = 0; i < this->size.first; i++)
        {
            delete[] added[i];
        }
        delete[] added;

        return output;
    }
}

template <typename T>
Tensor<T> Tensor<T>::OMPscalarMultiply(float multiplicand)
{
    T **multiplied = new T *[this->size.first];
    pair<int, int> ml_size = make_pair(this->size.first, this->size.second);
    for (int i = 0; i < this->size.first; i++)
    {
        multiplied[i] = new T[this->size.second];
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < this->size.first; i++)
    {
        for (int j = 0; j < this->size.second; j++)
        {
            multiplied[i][j] = this->data[i][j] * multiplicand;
        }
    }

    Tensor<T> output(multiplied, ml_size);

    for (int i = 0; i < this->size.first; i++)
    {
        delete[] multiplied[i];
    }
    delete[] multiplied;

    return output;
}

template <typename T>
Tensor<T> Tensor<T>::divide(Tensor<T> divisor)
{
    if (this->size.first != divisor.size.first || this->size.second != divisor.size.second)
        throw std::invalid_argument("Matrices should be of same dimension (a,b) / (a,b)");

    T **divided;
    pair<int, int> ml_size = make_pair(this->size.first, this->size.second);

    if (this->device_count > 0)
    {
        T **device_data_divided;
        CUDAdivide<T>(this->device_data, divisor.device_data, device_data_divided, this->size);

        Tensor<T> output(ml_size, 0);
        output.device_data = device_data_divided;

        return output;
    }
    else
    {
        divided = new T *[this->size.first];
        for (int i = 0; i < this->size.first; i++)
        {
            divided[i] = new T[this->size.second];
            for (int j = 0; j < this->size.second; j++)
            {
                divided[i][j] = this->data[i][j] / divisor.data[i][j];
            }
        }

        Tensor<T> output(divided, ml_size);

        for (int i = 0; i < this->size.first; i++)
        {
            delete[] divided[i];
        }
        delete[] divided;

        return output;
    }
}

template <typename T>
Tensor<T> Tensor<T>::add(Tensor<T> adder)
{
    if (this->size.first != adder.size.first || this->size.second != adder.size.second)
        throw std::invalid_argument("Matrices should be of same dimension (a,b) + (a,b)");

    T **added;
    pair<int, int> ml_size = make_pair(this->size.first, this->size.second);

    if (this->device_count > 0)
    {
        T **device_data_added;
        CUDAadd<T>(this->device_data, adder.device_data, device_data_added, this->size);

        Tensor<T> output(ml_size, 0);
        output.device_data = device_data_added;

        return output;
    }
    else
    {
        added = new T *[this->size.first];
        for (int i = 0; i < this->size.first; i++)
        {
            added[i] = new T[this->size.second];
            for (int j = 0; j < this->size.second; j++)
            {
                added[i][j] = this->data[i][j] + adder.data[i][j];
            }
        }

        Tensor<T> output(added, ml_size);

        for (int i = 0; i < this->size.first; i++)
        {
            delete[] added[i];
        }
        delete[] added;

        return output;
    }
}

template <typename T>
Tensor<T> Tensor<T>::elem_multiply(Tensor<T> multiplier)
{
    if (this->size.first != multiplier.size.first || this->size.second != multiplier.size.second)
        throw std::invalid_argument("Matrices should be of same dimension (a,b) * (a,b)");

    T **multiplied;
    pair<int, int> ml_size = make_pair(this->size.first, this->size.second);

    if (this->device_count > 0)
    {
        T **device_data_multiplied;
        CUDAelemMultiply<T>(this->device_data, multiplier.device_data, device_data_multiplied, this->size);

        Tensor<T> output(ml_size, 0);
        output.device_data = device_data_multiplied;

        return output;
    }
    else
    {
        multiplied = new T *[this->size.first];
        for (int i = 0; i < this->size.first; i++)
        {
            multiplied[i] = new T[this->size.second];
            for (int j = 0; j < this->size.second; j++)
            {
                multiplied[i][j] = this->data[i][j] * multiplier.data[i][j];
            }
        }

        Tensor<T> output(multiplied, ml_size);

        for (int i = 0; i < this->size.first; i++)
        {
            delete[] multiplied[i];
        }
        delete[] multiplied;

        return output;
    }
}

template <typename T>
Tensor<T> Tensor<T>::OMPadd(Tensor<T> adder)
{
    if (this->size.first != adder.size.first || this->size.second != adder.size.second)
        throw std::invalid_argument("Matrices should be of same dimension (a,b) + (a,b)");

    T **added = new T *[this->size.first];
    pair<int, int> ml_size = make_pair(this->size.first, this->size.second);
    for (int i = 0; i < this->size.first; i++)
    {
        added[i] = new T[this->size.second];
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < this->size.first; i++)
    {
        for (int j = 0; j < this->size.second; j++)
        {
            added[i][j] = this->data[i][j] + adder.data[i][j];
        }
    }
    Tensor<T> output(added, ml_size);

    for (int i = 0; i < this->size.first; i++)
    {
        delete[] added[i];
    }
    delete[] added;

    return output;
}

template <typename T>
Tensor<T> Tensor<T>::copy()
{
    return Tensor<T>(this->data, this->size);
}

template <typename T>
Tensor<float> Tensor<T>::convertFloat()
{
    float **floatTensor;

    if (this->device_count > 0)
    {
        T **floatTensor;
        CUDAconvertFloat<T>(this->device_data, floatTensor, this->size);

        Tensor<float> output(floatTensor, this->size);
        output.device_data = floatTensor;

        return output;
    }
    else
    {
        floatTensor = new float *[this->size.first];
        for (int i = 0; i < this->size.first; i++)
        {
            floatTensor[i] = new float[this->size.second];
            for (int j = 0; j < this->size.second; j++)
            {
                floatTensor[i][j] = float(this->data[i][j]);
            }
        }

        Tensor<float> output(floatTensor, this->size);

        for (int i = 0; i < this->size.first; i++)
        {
            delete[] floatTensor[i];
        }
        delete[] floatTensor;

        return output;
    }
}

template <typename T>
Tensor<float> Tensor<T>::sqrt()
{
    float **floatTensor;

    if (this->device_count > 0)
    {
        T **sqrtTensor;
        CUDAsqrt<T>(this->device_data, sqrtTensor, this->size);

        Tensor<float> output(sqrtTensor, this->size);
        output.device_data = sqrtTensor;

        return output;
    }
    else
    {
        floatTensor = new float *[this->size.first];
        for (int i = 0; i < this->size.first; i++)
        {
            floatTensor[i] = new float[this->size.second];
            for (int j = 0; j < this->size.second; j++)
            {
                floatTensor[i][j] = float(std::sqrt(this->data[i][j]));
            }
        }

        Tensor<float> output(floatTensor, this->size);

        for (int i = 0; i < this->size.first; i++)
        {
            delete[] floatTensor[i];
        }
        delete[] floatTensor;

        return output;
    }
}

template <typename T>
void Tensor<T>::map(void (*func)(T *))
{
    if (device_count > 0)
    {
        this->moveToHost();
    }
    for (int i = 0; i < this->size.first; i++)
    {
        for (int j = 0; j < this->size.second; j++)
        {
            func(&this->data[i][j]);
        }
    }
    if (device_count > 0)
    {
        this->moveToDevice();
    }
}

template <typename T>
Tensor<T> Tensor<T>::flatten()
{
    T *flatTensor;

    if (this->device_count > 0)
    {
        CUDAflatten<T>(this->device_data, flatTensor, this->size);

        Tensor<T> output(make_pair(1, this->size.first * this->size.second), 0);

        T **device_data = new T *[1];
        device_data[0] = new T[this->size.first * this->size.second];
        memcpy(device_data[0], flatTensor, this->size.first * this->size.second * sizeof(T));
        output.device_data = device_data;

        return output;
    }
    else
    {
        T** data = new T*[1];
        data[0] = new T[this->size.first * this->size.second];

        int index = 0;
        for (int i = 0; i < this->size.first; i++)
        {
            for (int j = 0; j < this->size.second; j++)
            {
                data[0][index++] = this->data[i][j];
            }
        }

        Tensor<T> output(data, make_pair(1, this->size.first * this->size.second));

        delete[] data[0];
        delete[] data;

        return output;
    }
}

template <typename T>
T Tensor<T>::max()
{
    if (device_count > 0)
    {
        this->moveToHost();
    }
    T small = this->data[0][0];
    for (int i = 0; i < this->size.first; i++)
    {
        for (int j = 0; j < this->size.second; j++)
        {
            if (this->data[i][j] > small)
                small = this->data[i][j];
        }
    }

    return small;
}

template <typename T>
T Tensor<T>::min()
{
    if (device_count > 0)
    {
        this->moveToHost();
    }

    T big = this->data[0][0];

    for (int i = 0; i < this->size.first; i++)
    {
        for (int j = 0; j < this->size.second; j++)
        {
            if (this->data[i][j] < big)
                big = this->data[i][j];
        }
    }

    return big;
}

template <typename T>
Tensor<int> Tensor<T>::randomTensor(std::pair<int, int> size)
{
    srand(time(NULL));
    int **data = new int *[size.first];
    for (int i = 0; i < size.first; i++)
    {
        data[i] = new int[size.second];
        for (int j = 0; j < size.second; j++)
        {
            data[i][j] = rand() - rand();
        }
    }

    Tensor<int> output = Tensor<int>(data, size);

    for (int i = 0; i < size.first; i++)
    {
        delete[] data[i];
    }
    delete[] data;
    return output;
}

template <typename T>
Tensor<int> Tensor<T>::randomTensor(std::pair<int, int> size, int min, int max)
{
    srand(time(NULL));
    int **data = new int *[size.first];
    for (int i = 0; i < size.first; i++)
    {
        data[i] = new int[size.second];
        for (int j = 0; j < size.second; j++)
        {
            data[i][j] = rand() % max + min;
        }
    }

    Tensor<int> output = Tensor<int>(data, size);

    for (int i = 0; i < size.first; i++)
    {
        delete[] data[i];
    }
    delete[] data;
    return output;
}

template <typename T>
Tensor<float> Tensor<T>::randomFloatTensor(std::pair<int, int> size)
{
    srand(time(NULL));
    float **data = new float *[size.first];
    for (int i = 0; i < size.first; i++)
    {
        data[i] = new float[size.second];
        for (int j = 0; j < size.second; j++)
        {
            data[i][j] = (rand() - rand()) / RAND_MAX;
        }
    }

    Tensor<float> output = Tensor(data, size);

    for (int i = 0; i < size.first; i++)
    {
        delete[] data[i];
    }
    delete[] data;
    return output;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor &other)
{
    if (this != &other)
    {
        if (device_data != nullptr)
        {
            this->moveToHost();
        }

        for (int i = 0; i < size.first; ++i)
        {
            delete[] data[i];
        }
        delete[] data;

        size = other.size;
        device_count = other.device_count;
        if (other.device_count > 0)
        {
            device_data = other.device_data;
        }

        data = new T *[size.first];
        for (int i = 0; i < size.first; ++i)
        {
            data[i] = new T[size.second];
            std::copy(other.data[i], other.data[i] + size.second, data[i]);
        }

        if (device_count > 0)
        {
            this->moveToDevice();
        }
    }
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor other)
{
    return this->multiply(other);
}

template <typename T>
Tensor<T> Tensor<T>::operator+(Tensor other)
{
    return this->add(other);
}

template <typename T>
Tensor<T> Tensor<T>::operator-(Tensor other)
{
    return this->add(other.scalarMultiply(-1));
}

template <typename T>
Tensor<T>::Tensor(const Tensor &other) : size(other.size), device_count(other.device_count)
{
    data = new T *[size.first];
    for (int i = 0; i < size.first; ++i)
    {
        data[i] = new T[size.second];
        std::copy(other.data[i], other.data[i] + size.second, data[i]);
    }

    if (device_count > 0)
    {
        moveToDevice();
    }
}

template <typename T>
Tensor<T> Tensor<T>::reshape(pair<int, int> size)
{
    if (this->size.first * this->size.second != size.first * size.second)
    {
        throw std::runtime_error("Invalid shape, product of size values must match each other");
    }

    if (device_count > 0)
    {
        T **device_data_reshaped;
        CUDAreshape<T>(this->device_data, device_data_reshaped, this->size, size);

        Tensor<T> output(size, 0);
        output.device_data = device_data_reshaped;

        return output;
    }

    T **data = new T *[size.first];
    for (int i = 0; i < size.first; i++)
    {
        data[i] = new T[size.second];
    }

    Tensor<T> temp = this->flatten();

    int curr_row = 0, curr_col = 0;
    for (int i = 0; i < temp.getSize().second; i++)
    {
        if (curr_col != 0 && curr_col % size.second == 0)
        {
            curr_row += 1;
            curr_col = 0;
        }
        data[curr_row][curr_col] = temp.data[0][i];
        curr_col += 1;
    }

    Tensor<T> output(data, size);

    for (int i = 0; i < size.first; i++)
    {
        delete[] data[i];
    }
    delete[] data;

    return output;
}

template <typename T>
std::pair<int, int> Tensor<T>::argmax()
{
    if (device_count > 0)
    {
        this->moveToHost();
    }
    T max = this->data[0][0];
    int i_max = 0, j_max = 0;
    for (int i = 0; i < this->getSize().first; i++)
    {
        for (int j = 0; j < this->getSize().second; j++)
        {
            if (this->data[i][j] > max)
            {
                max = this->data[i][j];
                i_max = i;
                j_max = j;
            }
        }
    }

    return make_pair(i_max, j_max);
}

template <typename T>
std::pair<int, int> Tensor<T>::argmin()
{
    if (device_count > 0)
    {
        this->moveToHost();
    }
    T min = this->data[0][0];
    int i_min = 0, j_min = 0;
    for (int i = 0; i < this->getSize().first; i++)
    {
        for (int j = 0; j < this->getSize().second; j++)
        {
            if (this->data[i][j] > min)
            {
                min = this->data[i][j];
                i_min = i;
                j_min = j;
            }
        }
    }

    return make_pair(i_min, j_min);
}

template <typename T>
Tensor<T> Tensor<T>::readCSV(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file");
    }

    std::vector<std::vector<T>> data;

    std::string line;
    while (std::getline(file, line))
    {
        std::vector<T> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            if (cell.empty())
            {
                row.push_back(std::numeric_limits<T>::quiet_NaN()); // Push NaN for empty cell
            }
            else
            {
                std::stringstream converter(cell);
                T value;
                converter >> value;
                row.push_back(value);
            }
        }
        data.push_back(row);
    }

    if (data.empty())
    {
        std::cerr << "File is empty: " << filename << std::endl;
    }

    int num_rows = data.size();
    int num_cols = data[0].size();

    for (auto row_ : data)
    {
        if (row_.size() != num_cols)
            throw std::runtime_error("All rows must be of equal length");
    }

    std::pair<int, int> size = make_pair(num_rows, num_cols);

    T **data_tensor = new T *[size.first];

    for (int i = 0; i < size.first; i++)
    {
        data_tensor[i] = new T[size.second];
        for (int j = 0; j < size.second; j++)
        {
            data_tensor[i][j] = data[i][j];
        }
    }

    Tensor<T> out = Tensor(data_tensor, size);

    for (int i = 0; i < size.first; i++)
    {
        delete[] data_tensor[i];
    }

    delete[] data_tensor;

    return out;
}

template <typename T>
vector<Tensor<T>> Tensor<T>::row_split()
{
    vector<Tensor<T>> split;
    for (int i = 0; i < this->getSize().first; i++)
    {
        T **temp_data = new T *[1];
        temp_data[0] = new T[this->getSize().second];

        for (int j = 0; j < this->getSize().second; j++)
        {
            temp_data[0][j] = this->data[i][j];
        }

        Tensor<T> temp = Tensor<T>(temp_data, make_pair(1, this->getSize().second));
        split.push_back(temp);
        delete[] temp_data[0];
        delete[] temp_data;
    }
    return split;
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> Tensor<T>::input_output_split(vector<int> output_indices)
{
    for (int i : output_indices)
    {
        if (i >= this->getSize().second)
            throw std::runtime_error("indices not in range of tensor");
    }
    T **input_data = new T *[this->getSize().first];
    T **output_data = new T *[this->getSize().first];

    for (int i = 0; i < this->getSize().first; i++)
    {
        input_data[i] = new T[this->getSize().second - output_indices.size()];
        output_data[i] = new T[output_indices.size()];

        int inp_count = 0;
        int out_count = 0;

        for (int j = 0; j < this->getSize().second; j++)
        {
            if (Tensor::find(output_indices, j))
            {
                output_data[i][out_count] = this->data[i][j];
                out_count++;
            }
            else
            {
                input_data[i][inp_count] = this->data[i][j];
                inp_count++;
            }
        }
    }

    Tensor<T> input = Tensor<T>(input_data, make_pair(this->getSize().first, this->getSize().second - output_indices.size()));
    Tensor<T> output = Tensor<T>(output_data, make_pair(this->getSize().first, output_indices.size()));

    for (int i = 0; i < this->getSize().first; i++)
    {
        delete[] input_data[i];
        delete[] output_data[i];
    }
    delete[] input_data;
    delete[] output_data;

    return make_pair(input, output);
}

template <typename T>
bool Tensor<T>::find(vector<int> indices, int value)
{
    for (int i : indices)
    {
        if (value == i)
            return true;
    }
    return false;
}

template <typename T>
Tensor<T> Tensor<T>::Normalize()
{
    if (device_count > 0)
    {
        this->moveToHost();
    }
    Tensor<T> transposed_tensor = this->copy();
    transposed_tensor.transpose();

    for (int i = 0; i < transposed_tensor.getSize().first; i++)
    {
        T max = transposed_tensor.data[i][0];
        T min = transposed_tensor.data[i][0];
        for (int j = 0; j < transposed_tensor.getSize().second; j++)
        {
            if (transposed_tensor.data[i][j] > max)
                max = transposed_tensor.data[i][j];
            if (transposed_tensor.data[i][j] < min)
                min = transposed_tensor.data[i][j];
        }

        for (int j = 0; j < transposed_tensor.getSize().second; j++)
        {
            transposed_tensor.data[i][j] = (transposed_tensor.data[i][j] - min) / (max - min);
        }
    }
    transposed_tensor.transpose();
    return transposed_tensor;
}

template <typename T>
void Tensor<T>::moveToDevice()
{
    if (device_data != nullptr)
    {
        throw std::runtime_error("Data is already on the device");
    }

    device_data = new T *[size.first];
    for (int i = 0; i < size.first; ++i)
    {
        cudaMalloc(&device_data[i], size.second * sizeof(T));
        cudaMemcpy(device_data[i], data[i], size.second * sizeof(T), cudaMemcpyHostToDevice);
    }
}

template <typename T>
void Tensor<T>::moveToHost()
{
    if (device_data == nullptr)
    {
        throw std::runtime_error("No data on the device to move to host");
    }

    for (int i = 0; i < size.first; ++i)
    {
        cudaMemcpy(data[i], device_data[i], size.second * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(device_data[i]);
    }

    delete[] device_data;
    device_data = nullptr;
}
#endif
