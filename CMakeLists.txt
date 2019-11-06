cmake_minimum_required(VERSION 3.7.2)


find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Werror -std=c++17")


set(SOURCE_LIB src/dnnl.cpp src/layers/fullyconnected_layer.cpp src/layers/layer_base.cpp)
set(TBBROOT "/export/users/cache/tbb/lnx/2017.2.132/tbb")
set(DAALROOT "/nfs/inn/proj/numerics1/Users/egorsmir/daal/daal_versions/daal_2019u4/__release_lnx/daal/lib/intel64_lin/")

include_directories(include)
include_directories(${TBBROOT}/include/tbb)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib")

add_library(dnnl SHARED ${SOURCE_LIB})
