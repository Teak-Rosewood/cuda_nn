#include <iostream>
#include "Tensor.hpp"
#include <utility>
#include <cstdlib>

using namespace std;

int main()
{
    int **arr;
    pair<int,int> size(4,4);

    arr = new int*[size.first];
    for (int i = 0; i < size.first; ++i) {
        arr[i] = new int[size.second];
    }

    for(int i=0;i<size.first;i++)
    {
        for(int j=0;j<size.second;j++)
        {
            arr[i][j] = rand();
        }
    }

    Tensor<int> a(arr,size);
    // a.print();
    // Tensor b = a.copy();
    // b.print();
    // b.print();
    // cout<<"\n";
    // a.OMPtranspose();
    // a.print();
    // cout<<endl;
    Tensor b = a.convertFloat().scalarMultiply(1.2);
    b.print();
    b.inverse();
    b.print();
}