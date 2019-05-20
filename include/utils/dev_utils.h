#ifndef __DEV_UTILS_H__
#define __DEV_UTILS_H__

#include <assert.h>
#include <iostream>

#define ASSERT_TRUE(__ASSERT_VALUE) \
    assert((__ASSERT_VALUE));

template<typename T>
void printArr(const T* data, size_t n, std::string str = std::string()) {
    std::cout << str <<  ": [ ";
    for(size_t i = 0; i < n; ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << " ]" << std::endl;
}

template<typename T>
void printArr(const T* data, size_t dim1, size_t dim2, std::string str = std::string()) {
    std::cout << str <<  ": [ ";
    for(size_t i = 0; i <  dim1; ++i) {
        std::cout << "[";
        for(size_t j = 0; j < dim2; ++j) {
            std::cout << data[i * dim2 + j] << ", ";
        }
        std::cout << "], ";
    }
    std::cout << " ]" << std::endl;
}

#endif // __DEV_UTILS_H__