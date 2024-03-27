cmake_minimum_required(VERSION 3.18)
project(arachne)

set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "-g")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_75;-O3;-Xcompiler -fPIC)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(SWIG REQUIRED)
find_package(PythonLibs 3.9 REQUIRED)

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
include(${SWIG_USE_FILE})

# Specify source and header files for the shared library
file(GLOB_RECURSE CUDA_SOURCES "include/*.cu")
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_HEADERS "include/*.hpp")

set(SOURCES ${CUDA_SOURCES} ${CPP_SOURCES})
set(HEADERS ${CUDA_HEADERS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

cuda_add_library(arachne SHARED ${SOURCES} ${HEADERS})
target_link_libraries(arachne ${CUDA_LIBRARIES})

# if(OpenMP_CXX_FOUND)
#     target_compile_options(arachne PRIVATE -fopenmp)
#     target_link_libraries(arachne OpenMP::OpenMP_CXX)
# endif()

set_target_properties(arachne PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Specify source and header files for the executables
# file(GLOB_RECURSE MAIN_SOURCES "src/model.cu" "src/main.cpp")
file(GLOB_RECURSE TEST_SOURCES "src/test.cpp")

# cuda_add_executable(main ${HEADERS} ${MAIN_SOURCES})
cuda_add_executable(test ${TEST_SOURCES})
target_link_libraries(test arachne)
set_target_properties(test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_SWIG_FLAGS "")
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_SOURCE_DIR})
set_property(SOURCE interfaces/arachne.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE interfaces/arachne.i PROPERTY SWIG_OUTFILE_DIR ${CMAKE_SOURCE_DIR})
swig_add_library(arachne_nn LANGUAGE python SOURCES interfaces/arachne.i  ${HEADERS})
swig_link_libraries(arachne_nn ${PYTHON_LIBRARIES} arachne ${CUDA_LIBRARIES})
set_target_properties(arachne_nn PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
