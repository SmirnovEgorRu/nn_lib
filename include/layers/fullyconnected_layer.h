
#ifndef __FULLYCONNECTED_LAYER_H__
#define __FULLYCONNECTED_LAYER_H__

#include "layer_base.h"

template<typename InType, typename OutType>
class Fullyconnected: public LayerBase<InType, OutType> {
    using LayerBase<InType, OutType>::_weights;
    using LayerBase<InType, OutType>::_biases;
    using LayerBase<InType, OutType>::_weightsDerivatives;
    using LayerBase<InType, OutType>::_biasesDerivatives;
    using LayerBase<InType, OutType>::_output;
    using LayerBase<InType, OutType>::_gradient;
public:

    Fullyconnected(size_t nInputNeurons, size_t nOutputNeurons, Initializer<InType> initWeigts = Initializer<InType>(), Initializer<InType> initBiases = Initializer<InType>());

    virtual void forward(const Tensor<InType>& data);
    virtual void backward();
    virtual void backward(const Tensor<InType>& grad, bool compute_grad = true);
    void computeWeightsDerivatives(size_t batchSize, const Tensor<InType>& grad);

protected:
    const Tensor<InType>* _data;

    size_t _nInputNeurons;
    size_t _nOutputNeurons;
    Initializer<InType> _initWeigts;
    Initializer<InType> _initBiases;
};

#endif // __FULLYCONNECTED_LAYER_H__
