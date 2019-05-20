
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

    Fullyconnected(size_t nInputNeurons, size_t nOutputNeurons, Initializer<InType> initWeigts = Initializer<InType>(), Initializer<InType> initBiases = Initializer<InType>()):
    _nInputNeurons(nInputNeurons), _nOutputNeurons(nOutputNeurons), _initWeigts(initWeigts), _initBiases(initBiases) {
        _weights.resize(size_t(nOutputNeurons), size_t(nInputNeurons));
        _biases.resize(size_t(1), size_t(1));
        _weightsDerivatives.resize(size_t(nOutputNeurons), size_t(nInputNeurons));
        _biasesDerivatives.resize(size_t(1), size_t(1));

        _initWeigts.generate(_weights.data(), nOutputNeurons*nInputNeurons);
        _initBiases.generate(_biases.data(), 1);
    }

    virtual void forward(const Tensor<InType>& data) {
        auto [in1, in2] = data.getFrameSize();
        ASSERT_TRUE(_nInputNeurons == in1 * in2);

        _data = &data;
        size_t BatchSize = data.getNBatches();
        _output.resize(BatchSize, _nOutputNeurons, 1);

        // out[batchSize, n_out] =  x[batchSize, n_in] * w[n_out, n_in]^T
        gemm(false, true, BatchSize, _nOutputNeurons, _nInputNeurons, 1.0f, data.data(), _nInputNeurons, _weights.data(), _nInputNeurons, 0.0f, _output.data(), _nOutputNeurons);
    }

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

    virtual void backward(const Tensor<InType>& grad, bool compute_grad = true) {
        auto [grd1, grd2] = grad.getFrameSize();
        size_t batchSize = grad.getNBatches();
        ASSERT_TRUE(grd1*grd2 == _nOutputNeurons);
        ASSERT_TRUE(batchSize == _data->getNBatches());

        _gradient.resize(batchSize, _nInputNeurons, 1);

        if (compute_grad) {
            // g[batchSize, n_in] = next_sigma[batchSize, n_out] * w[n_out, n_in]
            gemm(false, false, batchSize, _nInputNeurons, _nOutputNeurons, 1.0f, grad.data(), _nOutputNeurons, _weights.data(), _nInputNeurons, 0.0f, _gradient.data(), _nInputNeurons);
        }
        computeWeightsDerivatives(batchSize, grad);
    }

    void computeWeightsDerivatives(size_t batchSize, const Tensor<InType>& grad) {
        InType alpha = (1.0f / InType(batchSize));

        // DO: dw[n_out, n_in] = next_sigma[batchSize, n_out]^T * data[batchSize, n_in]
        gemm(true, false, _nOutputNeurons, _nInputNeurons, batchSize, alpha, grad.data(), _nOutputNeurons, _data->data(), _nInputNeurons, 0.0f, _weightsDerivatives.data(), _nInputNeurons);
    }

protected:
    const Tensor<InType>* _data;

    size_t _nInputNeurons;
    size_t _nOutputNeurons;
    Initializer<InType> _initWeigts;
    Initializer<InType> _initBiases;
};

#endif // __FULLYCONNECTED_LAYER_H__
