
#ifndef __SOFTMAX_LAYER_H__
#define __SOFTMAX_LAYER_H__

#include <cmath>
#include <omp.h>
#include "layer_base.h"
#include "utils/blas.h"


template<typename InType, typename OutType>
class SoftMaxLayer: public LayerBase<InType, OutType> {
    using LayerBase<InType, OutType>::_weights;
    using LayerBase<InType, OutType>::_biases;
    using LayerBase<InType, OutType>::_weightsDerivatives;
    using LayerBase<InType, OutType>::_biasesDerivatives;
    using LayerBase<InType, OutType>::_output;
    using LayerBase<InType, OutType>::_gradient;
public:
    SoftMaxLayer() {
    }

    InType sigm(InType val) {
        if (val >= 0.0f) {
            return 1.0f / (1.0f + std::exp(-val));
        } else {
            return std::exp(val) / (1 + std::exp(val));
        }
    }

    virtual void forward(const Tensor<InType>& data) {
        auto [in1, in2] = data.getFrameSize();
        _output.resize(data.getDims().begin(), data.getDims().end());

        const InType* x = data.data();
        InType* res = _output.data();

        size_t nBatches = data.getNBatches();

        size_t size = data.size();

        for(size_t i = 0; i < size; ++i) {
            res[i] = x[i];
            if (res[i] < -75.f)
                res[i] = -75.f;
            if (res[i] > 75.0f)
                res[i] = 75.f;
        }
        NnExp(size, res, res);

        for(size_t j = 0; j < nBatches; ++j) {
            InType sum = 0;
            for(size_t i = (in1*in2)*j; i < (in1*in2)*(j+1); ++i) {
                sum += res[i];
            }
            for(size_t i = (in1*in2)*j; i < (in1*in2)*(j+1); ++i) {
                res[i] = res[i] / sum;
            }
        }

        _data = &data;
    }

    virtual void backward(const Tensor<InType>& grad, bool compute_grad = true) {
        auto [in1, in2] = _data->getFrameSize();
        size_t size = _data->size();
        ASSERT_TRUE(size == grad.size());
        ASSERT_TRUE(size == _output.size());

        _weightsDerivatives.resize(size_t(1), size_t(1));
        _weights.resize(size_t(1), size_t(1));

        if(compute_grad) {
            _gradient.resize(grad.getDims().begin(), grad.getDims().end());
            const InType* out = _output.data();
            InType* res = _gradient.data();
            const InType* delta = grad.data();

            size_t nt = std::min<size_t>(size / 512, omp_get_max_threads());
            #pragma omp parallel for if(nt>1) num_threads(nt)
            for(size_t i = 0; i < size; ++i) {
                res[i] = delta[i] * out[i] * (1.0f - out[i]);
            }
        }
    }
protected:
    const Tensor<InType>* _data;
};

#endif // __SOFTMAX_LAYER_H__
