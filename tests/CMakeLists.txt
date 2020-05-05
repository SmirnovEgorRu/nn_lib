cmake_minimum_required(VERSION 3.7.2)

# Download and install GoogleTest
include(ExternalProject)
ExternalProject_Add(
    gtest
    URL https://github.com/google/googletest/archive/master.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(gtest source_dir binary_dir)
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/lib/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)
add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gtest)
set_target_properties(libgmock PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googlemock/libgmock.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)
include_directories("${source_dir}/googletest/include"
                    "${source_dir}/googlemock/include")

# Setup MKL and TBB
set(TBBROOT $ENV{TBBROOT})
set(MKLROOT $ENV{MKLROOT})
message(TBBROOT=${TBBROOT})
message(MKLROOT=${MKLROOT})
include_directories(${TBBROOT}/include/tbb)

# Add include dirs
include_directories("../../include"
                    "../include")

# Setup openMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add GCC flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Werror -std=c++17 -march=cascadelake -lpthread -lm ")

add_executable(tests main.cpp)

# Link libs
target_link_libraries(tests libgtest)
target_link_libraries(tests ${TBBROOT}/lib/intel64/gcc4.8/libtbb.so.2)
target_link_libraries(tests ${TBBROOT}/lib/intel64/gcc4.8/libtbbmalloc.so.2)
target_link_libraries(tests ${MKLROOT}/lib/intel64/libmkl_intel_lp64.so)
target_link_libraries(tests ${MKLROOT}/lib/intel64/libmkl_core.so)
target_link_libraries(tests ${MKLROOT}/lib/intel64/libmkl_sequential.so)
target_link_libraries(tests ${CMAKE_CURRENT_SOURCE_DIR}/../build/lib/libdnnl.so)
