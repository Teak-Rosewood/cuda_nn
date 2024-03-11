cmake_minimum_required(VERSION 3.18)
project(cuda_nn)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(SWIG REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_PATH})
include(${SWIG_USE_FILE})

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

set(CMAKE_SWIG_FLAGS "")
set_property(SOURCE interfaces/cuda_nn.i PROPERTY CPLUSPLUS ON)
swig_add_library(cudann LANGUAGE python SOURCES interfaces/cuda_nn.i ${HEADERS})
swig_link_libraries(cudann ${PYTHON_LIBRARIES})
set_target_properties(cudann PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)