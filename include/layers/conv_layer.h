
#ifndef __CONV_LAYER_H__
#define __CONV_LAYER_H__

#include "layer_base.h"

static bool COMPUTE_IN_INT8 = false;

template<typename AType, typename ATypeTmp, typename BType, typename BTypeTmp, typename CType>
void _conv_simple(const AType* frame, const BType* kernel, CType* res, size_t in1, size_t in2, size_t _k1, size_t _k2) {
    size_t out1 = in1 - _k1 + 1;
    size_t out2 = in2 - _k2 + 1;

    for (size_t i = 0; i < out1; ++i) {
        for (size_t j = 0; j < out2; ++j) {
            CType sum = 0;
            for (size_t k1 = 0; k1 < _k1; ++k1) {
                for (size_t k2 = 0; k2 < _k2; ++k2) {
                    sum += ATypeTmp(frame[(i + k1)*in2 + j + k2]) * BTypeTmp(kernel[k1*_k2 + k2]);
                }
            }
            res[i * out2 + j] = sum;
        }
    }
}

void conv_simple(const float* frame, const float* kernel, float* res, size_t in1, size_t in2, size_t _k1, size_t _k2);
void vnni_conv_common(const uint8_t* frame, const int8_t* kernel, int32_t* res, int in1, int in2, int _k1, int _k2);

template<int _k2, int k4, int tail_k>
void vnni_conv_common_jit(const uint8_t* frame, const int8_t* kernel, int32_t* res, int in1, int in2, int _k1);
void vnni_conv_k1_1(const uint8_t* frame, const int8_t* kernel, int32_t* res, int in1, int in2, int _k1, int _k2);
void vnni_conv_k0_1(const uint8_t* frame, const int8_t* kernel, int32_t* res, int in1, int in2, int _k1, int _k2);

void vnni_conv(const uint8_t* frame, const int8_t* kernel, int32_t* res, int in1, int in2, int _k1, int _k2);

void _conv_blocking(const float* frame, const float* kernel, float* res, size_t in1, size_t in2, size_t _k1, size_t _k2);
void _conv_blocking_int8(const int8_t* frame, const int8_t* kernel, int32_t* res, size_t in1, size_t in2, size_t _k1, size_t _k2);
void _conv_avx512_if(const float* frame, const float* kernel, float* res, size_t in1, size_t in2, size_t _k1, size_t _k2);
void _conv_avx512(const float* frame, const float* kernel, float* res, int in1, int in2, int _k1, int _k2);
void _conv_avx512_null(const float* frame, const float* kernel, float* res, int in1, int in2, int _k1, int _k2);


size_t vnni_conv_get_buffer_size(const uint8_t* frame, const int8_t* kernel, int32_t* res, int in1, int in2, int _k1, int _k2);
void vnni_conv_fill_buffer(const uint8_t* frame, const int8_t* kernel, int32_t* res, int in1, int in2, int _k1, int _k2, uint8_t* buffer);
void vnni_conv_with_buffer(const uint8_t* buffer, const int8_t* kernel, int32_t* res, int in1, int in2, int _k1, int _k2);
void vnni_conv_buffer(const uint8_t* buffer, const int8_t* kernel, int32_t* res, int in1, int in2, int _k1, int _k2);

template<typename InType, typename OutType>
class Convolution2d: public LayerBase<InType, OutType> {
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

    Convolution2d(Parameter par, Initializer<InType> initWeigts = Initializer<InType>(), Initializer<InType> initBiases = Initializer<InType>());
    virtual void forward(const Tensor<InType>& data);
    virtual void backward(const Tensor<InType>& grad, bool compute_grad = true);
    void computeWeightsDerivatives(size_t batchSize, const Tensor<InType>& grad);

protected:
    const Tensor<InType>* _data;
    Parameter _par;
    size_t _in1;
    size_t _in2;
    size_t _out1;
    size_t _out2;
    size_t _k1;
    size_t _k2;

    Initializer<InType> _initWeigts;
    Initializer<InType> _initBiases;
};

#endif // __FULLYCONNECTED_LAYER_H__
