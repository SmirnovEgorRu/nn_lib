
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

    MaxPooling(Parameter par);
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
};

#endif // __FULLYCONNECTED_LAYER_H__
