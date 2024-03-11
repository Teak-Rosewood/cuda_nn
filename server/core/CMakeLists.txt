cmake_minimum_required(VERSION 3.18)
project(cuda_nn)

# Find CUDA package
find_package(CUDA REQUIRED)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Set C++ version
set(CMAKE_CXX_STANDARD 11)

# Specify source files
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")

# Set CUDA architecture (change it according to your GPU capabilities)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_35;-O3;-Xcompiler -fPIC)

# Specify include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compile CUDA sources to object files
cuda_compile(cuda_objs ${CUDA_SOURCES})

# Add the compiled CUDA objects and C++ sources to the executable
add_library(cuda_nn SHARED ${cuda_objs} ${CPP_SOURCES} ${CUDA_SOURCES})

# Link CUDA toolkit libraries
target_link_libraries(cuda_nn ${CUDA_LIBRARIES})

# Enable OpenMP support
if(OpenMP_CXX_FOUND)
    target_compile_options(cuda_nn PRIVATE -fopenmp)
    target_link_libraries(cuda_nn OpenMP::OpenMP_CXX)
endif()

# Set output directory for the shared library
set_target_properties(cuda_nn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

set (SOURCES
    include/model.hpp
    src/model.cu
    src/main.cpp
)

set (MLFile 
    include/Activation.hpp
    include/Activation.hpp        
    include/Linear.hpp
    include/MSELoss.hpp    
    include/Relu.hpp     
    include/Tensor.hpp
    include/Adam.hpp              
    include/Loss.hpp     
    include/Normalize.hpp  
    include/RMSProp.hpp  
    include/Variables.hpp
    include/CrossEntropyLoss.hpp  
    include/MAELoss.hpp  
    include/Optimizer.hpp  
    include/SGD.hpp
    include/Flatten.hpp
    include/Model.hpp    
    include/Pipeline.hpp   
    include/Softmax.hpp    
    src/test.cpp
)

cuda_add_executable(main ${SOURCES})
cuda_add_executable(test ${MLFile})
cuda_add_library(main_shared SHARED ${SOURCES})
cuda_add_library(test_shared SHARED ${MLFile})