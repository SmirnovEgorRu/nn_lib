
#ifndef __SIGMOID_LAYER_H__
#define __SIGMOID_LAYER_H__

#include <cmath>
#include <omp.h>
#include "layer_base.h"
#include "utils/blas.h"


template<typename InType, typename OutType>
class Sigmoid: public LayerBase<InType, OutType> {
    using LayerBase<InType, OutType>::_weights;
    using LayerBase<InType, OutType>::_biases;
    using LayerBase<InType, OutType>::_weightsDerivatives;
    using LayerBase<InType, OutType>::_biasesDerivatives;
    using LayerBase<InType, OutType>::_output;
    using LayerBase<InType, OutType>::_gradient;
public:
    Sigmoid() {
    }

    virtual void forward(const Tensor<InType>& data) {
        auto [in1, in2] = data.getFrameSize();
        _output.resize(data.getDims().begin(), data.getDims().end());

        const InType* x = data.data();
        InType* res = _output.data();

        size_t size = data.size();

        size_t ntr = std::min<size_t>(size / 512, omp_get_max_threads());
        #pragma omp parallel if(ntr>1) num_threads(ntr)
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();

            int blockSize = size / nt + !!(size % nt);
            int iStart = tid * blockSize;
            int iEnd   = (tid+1) * blockSize > size ? size : (tid+1) * blockSize;

            for(size_t i = iStart; i < iEnd; ++i) {
                res[i] = -x[i];
                if (res[i] < -75.f)
                    res[i] = -75.f;
                if (res[i] > 75.0f)
                    res[i] = 75.f;
            }
            NnExp(iEnd - iStart, res + iStart, res + iStart);
            for(size_t i = iStart; i < iEnd; ++i) {
                res[i] = 1.0f / (1.0f + res[i]);
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

#endif // __SIGMOID_LAYER_H__
