
#ifndef __LAYER_BASE_H__
#define __LAYER_BASE_H__

#include "utils/tensor.h"
#include "model/initializers.h"
#include "utils/blas.h"

template<typename InType, typename OutType>
class LayerBase {
public:
    virtual Tensor<InType>& getWeightsDerivatives() {
        return _weightsDerivatives;
    }
    virtual Tensor<InType>& getBiasDerivatives() {
        return _biasesDerivatives;
    }

    virtual Tensor<InType>& getOutput() {
        return _output;
    }

    virtual Tensor<InType>& getGradient() {
        return _gradient;
    }

    virtual Tensor<InType>& getWeights() {
        return _weights;
    }

    virtual Tensor<InType>& getBiases() {
        return _biases;
    }

    void setGroundTruth(Tensor<InType>& groundTruth) {
        _groundTruth = &groundTruth;
    }

    virtual void forward(const Tensor<InType>& data) = 0;

    virtual void backward() {
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

    virtual void backward(const Tensor<InType>& grad, const bool compute_grad = true) = 0;

    LayerBase<InType, OutType>* prev = nullptr;
    LayerBase<InType, OutType>* next = nullptr;
protected:

    Tensor<InType> _weights;
    Tensor<InType> _biases;
    Tensor<InType> _weightsDerivatives;
    Tensor<InType> _biasesDerivatives;
    Tensor<InType> _output;
    Tensor<InType> _gradient;


    Tensor<InType>* _groundTruth = nullptr;
};

#endif // #ifndef __LAYER_BASE_H__
