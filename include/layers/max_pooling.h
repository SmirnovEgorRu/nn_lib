
#ifndef __MAX_POOLING_LAYER_H__
#define __MAX_POOLING_LAYER_H__

#include "layer_base.h"
#include "initializers.h"
#include "blas.h"

template<typename InType, typename OutType>
class MaxPooling: public LayerBase<InType, OutType> {
    using LayerBase<InType, OutType>::_weights;
    using LayerBase<InType, OutType>::_biases;
    using LayerBase<InType, OutType>::_weightsDerivatives;
    using LayerBase<InType, OutType>::_biasesDerivatives;
    using LayerBase<InType, OutType>::_output;
    using LayerBase<InType, OutType>::_gradient;
public:

    struct Parameter {
        size_t dim1;
        size_t dim2;
        size_t kernel_dim1;
        size_t kernel_dim2;
    };

    MaxPooling(Parameter par): _par(par) {

        _in1  = _par.dim1;
        _in2  = _par.dim2;
        _k1   = _par.kernel_dim1;
        _k2   = _par.kernel_dim2;
        _out1 = _in1 / _k1;
        _out2 = _in2 / _k2;

        _weights.resize(size_t(1), size_t(1));
        _biases.resize(size_t(1), size_t(1));

        _weightsDerivatives.resize(size_t(1), size_t(1));
        _biasesDerivatives.resize(size_t(1), size_t(1));
    }

    virtual void forward(const Tensor<InType>& data) {
        auto [in1, in2] = data.getFrameSize();
        ASSERT_TRUE(_in1 == in1);
        ASSERT_TRUE(_in2 == in2);

        _data = &data;
        size_t BatchSize = data.getNBatches();
        _output.resize(BatchSize, _out1, _out2);

        InType* w = _weights.data();

        #pragma omp parallel for
        for (size_t iBatch = 0; iBatch < BatchSize; ++iBatch) {
            InType* out =  _output.data() + iBatch* (_out1 * _out2);
            const InType* x   =  data.data() + iBatch* (_in1 * _in2);

            for (size_t i = 0; i < _out1; ++i) {
                for (size_t j = 0; j < _out2; ++j) {
                    InType sum = 0;
                    for (size_t k1 = 0; k1 < _k1; ++k1) {
                        for (size_t k2 = 0; k2 < _k2; ++k2) {
                            sum += x[(i + k1)*_in2 + j + k2] * w[k1*_k2 + k2];
                        }
                    }
                    out[i * _out2 + j] = sum;
                }
            }
        }
    }

    virtual void backward(const Tensor<InType>& grad, bool compute_grad = true) {
        auto [grd1, grd2] = grad.getFrameSize();
        size_t batchSize = grad.getNBatches();

        ASSERT_TRUE(grd1 * grd2 == _out1 * _out2);
        ASSERT_TRUE(batchSize == _data->getNBatches());

        if (compute_grad) {
            _gradient.resize(batchSize, _in1, _in2);
            InType* w = _weights.data();

            size_t tmp1 = _out1 + 2 * _k1 - 2;
            size_t tmp2 = _out2 + 2 * _k2 - 2;

            Tensor<InType> tmp(batchSize, tmp1, tmp2);

            #pragma omp parallel for
            for (size_t iBatch = 0; iBatch < batchSize; ++iBatch) {
                const InType* grNext  =  grad.data() + iBatch* (_out1 * _out2);
                InType* g       =  _gradient.data() + iBatch* (_in1 * _in2);
                InType* buff    = tmp.data() + iBatch * tmp1 * tmp2;

                std::fill_n(buff, tmp1 * tmp2, InType(0));
                for (size_t i = 0; i < _out1; ++i) {
                    for (size_t j = 0; j < _out2; ++j) {
                        buff[(i + _k1 -1)*tmp2 + j + _k2  - 1] = grNext[i*_out2 + j];
                    }
                }

                std::fill_n(g, _in1 * _in2, InType(0));
                for (size_t i = 0; i < _in1; ++i) {
                    for (size_t j = 0; j < _in2; ++j) {
                        InType sum = 0;
                        for (size_t k1 = 0; k1 < _k1; ++k1) {
                            for (size_t k2 = 0; k2 < _k2; ++k2) {
                                sum += buff[(i + k1)*tmp2 + j + k2] * w[k1*_k2 + k2];
                            }
                        }
                        g[i*_in2 + j] = sum;
                    }
                }
            }
        }

        computeWeightsDerivatives(batchSize, grad);
    }

    void computeWeightsDerivatives(size_t batchSize, const Tensor<InType>& grad) {
        InType* dw = _weightsDerivatives.data();
        std::fill_n(dw, _k1 * _k2, InType(0));

        for (size_t iBatch = 0; iBatch < batchSize; ++iBatch) {
            const InType* g =  grad.data() + iBatch * (_out1 * _out2);
            const InType* x  =  _data->data() + iBatch* (_in1 * _in2);

            for (size_t i = 0; i < _out1; ++i) {
                for (size_t j = 0; j < _out2; ++j) {
                    InType sum = 0;
                    for (size_t k1 = 0; k1 < _k1; ++k1) {
                        for (size_t k2 = 0; k2 < _k2; ++k2) {
                            dw[k1*_k2 + k2] +=  g[i * _out2 + j] * x[(i + k1)*_in2 + j + k2];
                        }
                    }
                }
            }
        }
        InType alpha = (1.0f / InType(batchSize));
        for (size_t k1 = 0; k1 < _k1; ++k1) {
            for (size_t k2 = 0; k2 < _k2; ++k2) {
                dw[k1*_k2 + k2] *=  alpha;
            }
        }
    }

protected:
    const Tensor<InType>* _data;
    Parameter _par;
    size_t _in1;
    size_t _in2;
    size_t _out1;
    size_t _out2;
    size_t _k1;
    size_t _k2;
};

#endif // __FULLYCONNECTED_LAYER_H__
