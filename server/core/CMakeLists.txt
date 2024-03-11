cmake_minimum_required(VERSION 3.18)
project(cuda_nn)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 11)

# Specify source and header files for the shared library
file(GLOB_RECURSE CUDA_SOURCES "src/.cu")
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_HEADERS "include/*.hpp")

set(SOURCES ${CUDA_SOURCES} ${CPP_SOURCES})
set(HEADERS ${CUDA_HEADERS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_35;-O3;-Xcompiler -fPIC)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

cuda_compile(cuda_objs ${CUDA_SOURCES})

add_library(cuda_nn SHARED ${cuda_objs} ${SOURCES})
target_link_libraries(cuda_nn ${CUDA_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_compile_options(cuda_nn PRIVATE -fopenmp)
    target_link_libraries(cuda_nn OpenMP::OpenMP_CXX)
endif()

set_target_properties(cuda_nn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Specify source and header files for the executables
# file(GLOB_RECURSE MAIN_SOURCES "src/model.cu" "src/main.cpp")
file(GLOB_RECURSE TEST_SOURCES "src/test.cpp")

# cuda_add_executable(main ${HEADERS} ${MAIN_SOURCES})
cuda_add_executable(test ${HEADERS} ${TEST_SOURCES})

set_target_properties(test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# cmake_minimum_required(VERSION 3.18)
# project(cuda_nn)

# # Find package
# find_package(CUDA REQUIRED)
# find_package(OpenMP REQUIRED)

# # Set C++ version
# set(CMAKE_CXX_STANDARD 11)

# # Specify source and header files
# file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
# file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
# file(GLOB_RECURSE CUDA_HEADERS "include/*.hpp")
# set(SOURCES ${CUDA_SOURCES} ${CPP_SOURCES})
# set(HEADERS ${CUDA_HEADERS})

# # Set CUDA architecture (change it according to your GPU capabilities)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_35;-O3;-Xcompiler -fPIC)

# # Specify include directories
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# # Compile CUDA sources to object files
# cuda_compile(cuda_objs ${CUDA_SOURCES}cmake_minimum_required(VERSION 3.18)
# project(cuda_nn)

# # Find package
# find_package(CUDA REQUIRED)
# find_package(OpenMP REQUIRED)

# # Set C++ version
# set(CMAKE_CXX_STANDARD 11)

# # Specify source and header files
# file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
# file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
# file(GLOB_RECURSE CUDA_HEADERS "include/*.hpp")
# set(SOURCES ${CUDA_SOURCES} ${CPP_SOURCES})
# set(HEADERS ${CUDA_HEADERS})

# # Set CUDA architecture (change it according to your GPU capabilities)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_35;-O3;-Xcompiler -fPIC)

# # Specify include directories
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# # Compile CUDA sources to object files
# cuda_compile(cuda_objs ${CUDA_SOURCES})

# # Add the compiled CUDA objects and C++ sources to the executable
# add_library(cuda_nn SHARED ${cuda_objs} ${SOURCES})

# # Link CUDA toolkit libraries
# target_link_libraries(cuda_nn ${CUDA_LIBRARIES})

# # Enable OpenMP support
# if(OpenMP_CXX_FOUND)
#     target_compile_options(cuda_nn PRIVATE -fopenmp)
#     target_link_libraries(cuda_nn OpenMP::OpenMP_CXX)
# endif()

# # Set output directory for the shared library
# set_target_properties(cuda_nn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# file(GLOB_RECURSE MAIN_SOURCES "src/model.cu" "src/main.cpp")
# file(GLOB_RECURSE TEST_SOURCES "src/test.cpp")

# cuda_add_executable(main ${HEADERS} ${MAIN_SOURCES})
# cuda_add_executable(test ${HEADERS} ${TEST_SOURCES}))

# # Add the compiled CUDA objects and C++ sources to the executable
# add_library(cuda_nn SHARED ${cuda_objs} ${SOURCES})

# # Link CUDA toolkit libraries
# target_link_libraries(cuda_nn ${CUDA_LIBRARIES})

# # Enable OpenMP support
# if(OpenMP_CXX_FOUND)
#     target_compile_options(cuda_nn PRIVATE -fopenmp)
#     target_link_libraries(cuda_nn OpenMP::OpenMP_CXX)
# endif()

# # Set output directory for the shared library
# set_target_properties(cuda_nn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# file(GLOB_RECURSE MAIN_SOURCES "src/model.cu" "src/main.cpp")
# file(GLOB_RECURSE TEST_SOURCES "src/test.cpp")

# cuda_add_executable(main ${HEADERS} ${MAIN_SOURCES})
# cuda_add_executable(test ${HEADERS} ${TEST_SOURCES})
# cuda_add_library(main_shared SHARED ${HEADERS} src/model.cu src/main.cpp)
# cuda_add_library(test_shared SHARED ${HEADERS} src/test.cpp)

# set_target_properties(main main_shared test test_shared
#     PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
#     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
# )
# set (SOURCES
#     include/model.hpp
#     src/model.cu
#     src/main.cpp
# )

# set (MLFile 
#     include/Activation.hpp
#     include/Activation.hpp        
#     include/Linear.hpp
#     include/MSELoss.hpp    
#     include/Relu.hpp     
#     include/Tensor.hpp
#     include/Adam.hpp              
#     include/Loss.hpp     
#     include/Normalize.hpp  
#     include/RMSProp.hpp  
#     include/Variables.hpp
#     include/CrossEntropyLoss.hpp  
#     include/MAELoss.hpp  
#     include/Optimizer.hpp  
#     include/SGD.hpp
#     include/Flatten.hpp
#     include/Model.hpp    
#     include/Pipeline.hpp   
#     include/Softmax.hpp    
#     src/test.cpp
# )

