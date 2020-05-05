cmake_minimum_required(VERSION 3.7.2)


find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Werror -std=c++17")


set(SOURCE_LIB src/dnnl.cpp src/layers/fullyconnected_layer.cpp src/layers/layer_base.cpp)

include_directories(include)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib")

add_library(dnnl SHARED ${SOURCE_LIB})
