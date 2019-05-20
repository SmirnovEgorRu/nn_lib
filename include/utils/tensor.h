

#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "dev_utils.h"
#include <vector>
#include <tuple>
#include "tbb.h"

template<typename T>
class Tensor {
public:
    template<typename Iter>
    Tensor(Iter begin, Iter end) {
        resize(begin, end);
    }

    Tensor(size_t d1, size_t d2) {
        resize(d1,d2);
    }

    Tensor(size_t d1, size_t d2, size_t d3) {
        resize(d1,d2,d3);
    }

    Tensor() {
    }

    template<typename Iter>
    void resize(Iter begin, Iter end) {
        if(end - begin == dims_.size()) {
            size_t size = 1;
            size_t i = 0;
            bool equal = true;
            for(auto it = begin; it != end; it++) {
                if (dims_[i++] != *it) equal = false;
            }
            if (equal) return;
        }

        dims_.clear();
        data_.clear();

        dims_.reserve(end - begin);
        size_t size = 1;
        for(; begin != end; begin++) {
            size *= (*begin);
            dims_.push_back(*begin);
        }
        data_.resize(size);
    }

    void resize(size_t d1, size_t d2) {
        if(dims_.size() == 2 &&
           dims_[0] == d1 &&
           dims_[1] == d2) {
            return;
        }

        dims_.clear();
        data_.clear();

        dims_.reserve(2);
        size_t size = d1 * d2;

        dims_.push_back(d1);
        dims_.push_back(d2);
        data_.resize(size);
    }

    void resize(size_t d1, size_t d2, size_t d3) {
        if(dims_.size() == 3 &&
           dims_[0] == d1 &&
           dims_[1] == d2 &&
           dims_[2] == d3) {
            return;
        }

        dims_.clear();
        data_.clear();

        dims_.reserve(3);
        size_t size = d1 * d2 * d3;

        dims_.push_back(d1);
        dims_.push_back(d2);
        dims_.push_back(d3);
        data_.resize(size);
    }

    const std::vector<size_t>& getDims() const {
        return dims_;
    }

    size_t getDim(size_t dim) const {
        ASSERT_TRUE(dims_.size() > dim);
        return dims_[dim];
    }

    const T* data() const {
        return data_.data();
    }

    T* data() {
        return data_.data();
    }

    size_t size() const {
        size_t size = 1;
        for(auto elem : dims_) {
            size *= (elem);
        }
        return size;
    }

    std::tuple<size_t, size_t> getFrameSize() const {
        ASSERT_TRUE(dims_.size() >= 1);
        if (dims_.size() == 1) {
            return std::make_tuple(dims_[0], 1);
        } else {
            return std::make_tuple(dims_[dims_.size()-2], dims_[dims_.size()-1]);
        }
    }

    size_t getNBatches() const {
        ASSERT_TRUE(dims_.size() >= 1);
        ASSERT_TRUE(dims_.size() <= 3);

        if (dims_.size() <= 2) {
            return 1;
        } else {
            return dims_[dims_.size()-3];
        }
    }

protected:
    std::vector<size_t> dims_;
    std::vector<T> data_;
};

#endif // #ifndef __TENSOR_H__
