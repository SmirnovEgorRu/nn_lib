#include "utils/tensor.h"
#include "model/initializers.h"
#include "utils/blas.h"
#include "layers/layer_base.h"

template class LayerBase<float, float>;

template<typename InType, typename OutType>
Tensor<InType>& LayerBase<InType, OutType>::getWeightsDerivatives() {
    return _weightsDerivatives;
}

template<typename InType, typename OutType>
Tensor<InType>& LayerBase<InType, OutType>::getBiasDerivatives() {
    return _biasesDerivatives;
}

template<typename InType, typename OutType>
Tensor<InType>& LayerBase<InType, OutType>::getOutput() {
    return _output;
}

template<typename InType, typename OutType>
Tensor<InType>& LayerBase<InType, OutType>::getGradient() {
    return _gradient;
}

template<typename InType, typename OutType>
Tensor<InType>& LayerBase<InType, OutType>::getWeights() {
    return _weights;
}

template<typename InType, typename OutType>
Tensor<InType>& LayerBase<InType, OutType>::getBiases() {
    return _biases;
}

template<typename InType, typename OutType>
void LayerBase<InType, OutType>::setGroundTruth(Tensor<InType>& groundTruth) {
    _groundTruth = &groundTruth;
}

template<typename InType, typename OutType>
void LayerBase<InType, OutType>::backward() {
    ASSERT_TRUE(this->_groundTruth != nullptr);
    ASSERT_TRUE(this->_groundTruth->size() == _output.size());
    Tensor<InType> grad(_output.getDims().begin(), _output.getDims().end());

    InType* gr = grad.data();
    InType* out = _output.data();
    InType* groundTruth = this->_groundTruth->data();

    for(size_t i = 0; i < grad.size(); ++i) {
        gr[i] = groundTruth[i] - out[i];
    }

    backward(grad, true);
}
