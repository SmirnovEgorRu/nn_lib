
#ifndef __LAYER_BASE_H__
#define __LAYER_BASE_H__

#include "utils/tensor.h"
#include "model/initializers.h"

template<typename InType, typename OutType>
class LayerBase {
public:
    virtual Tensor<InType>& getWeightsDerivatives();
    virtual Tensor<InType>& getBiasDerivatives();
    virtual Tensor<InType>& getOutput();
    virtual Tensor<InType>& getGradient();
    virtual Tensor<InType>& getWeights();
    virtual Tensor<InType>& getBiases();

    void setGroundTruth(Tensor<InType>& groundTruth);
    virtual void forward(const Tensor<InType>& data) = 0;
    virtual void backward();
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
