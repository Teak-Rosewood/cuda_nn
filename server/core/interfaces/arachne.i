// %module arachne
%module(directors="1") arachne

%{
#include <utility>
#include <cuda_runtime.h>
#include "kernels/kernels.hpp"
#include "Tensor.hpp"
#include "Model.hpp"
#include "Activation.hpp"
#include "Normalize.hpp"
#include "Optimizer.hpp"
#include "Adam.hpp"         
#include "Loss.hpp"
#include "CrossEntropyLoss.hpp"
#include "Pipeline.hpp"
#include "Flatten.hpp"
#include "Relu.hpp"
#include "RMSProp.hpp"
#include "Linear.hpp"
#include "SGD.hpp"
#include "Variables.hpp"
#include "Softmax.hpp"
#include "MAELoss.hpp"
#include "MSELoss.hpp"
%}
%include <std_string.i>
%include <std_vector.i>
%include <std_pair.i>
%include "kernels/kernels.hpp"

%template(CUDAscalarMultiplyFloat) CUDAscalarMultiply<float>;
%template(CUDAMultiplyFloat) CUDAMultiply<float>;
%template(CUDAscalarAddFloat) CUDAscalarAdd<float>;
%template(CUDAdivideFloat) CUDAdivide<float>;
%template(CUDAaddFloat) CUDAadd<float>;
%template(CUDAelemMultiplyFloat) CUDAelemMultiply<float>;
%template(CUDAflattenFloat) CUDAflatten<float>;
%template(CUDAreshapeFloat) CUDAreshape<float>;
%template(CUDAconvertFloatFloat) CUDAconvertFloat<float>;
%template(CUDAsqrtFloat) CUDAsqrt<float>;

%include "Tensor.hpp"
namespace std {
    %template(IntVector) vector<int>;
    %template(FloatTensorVector) vector<Tensor<float>>;
}
%template(FloatTensor) Tensor<float>;
namespace std {
    %template(FloatTensorPair) pair<Tensor<float>, Tensor<float>>;
    %template(IntPair) pair<int, int>;
}
%include "Model.hpp"
%include "Activation.hpp"
%include "Normalize.hpp"
%include "Optimizer.hpp"
%include "Adam.hpp"         
%include "Loss.hpp"
%include "CrossEntropyLoss.hpp"
%include "Pipeline.hpp"
%template(forwardFloat) Pipeline::forward<float>;
%include "Flatten.hpp"
%include "Relu.hpp"
%include "RMSProp.hpp"
%include "Linear.hpp"
%include "SGD.hpp"
%include "Variables.hpp"
%include "Softmax.hpp"
%include "MAELoss.hpp"
%include "MSELoss.hpp"
