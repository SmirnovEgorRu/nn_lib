
#ifndef __NN_TOPOLOGY_H__
#define __NN_TOPOLOGY_H__

#include "omp.h"

template<typename T>
class OptimizationsSolver {
public:
    virtual void solve(Tensor<T>& values, Tensor<T>& grad) = 0;
    virtual OptimizationsSolver<T>* clone() = 0;
};

template<typename T>
class SGDSolver: public OptimizationsSolver<T> {
public:

    SGDSolver(T learningRate) : learningRate_(learningRate) {
    }

    virtual void solve(Tensor<T>& values, Tensor<T>& grad) {
        size_t size = values.size();
        ASSERT_TRUE(size == grad.size());

        T* w = values.data();
        T* dw = grad.data();

        for(size_t j = 0; j < size; ++j) {
            w[j] += learningRate_ * dw[j];
        }
    }

    virtual OptimizationsSolver<T>* clone() {
        return new SGDSolver<T>(learningRate_);
    }

protected:
    T learningRate_;
};

template<typename T>
class SGDMomentumSolver: public OptimizationsSolver<T> {
public:

    SGDMomentumSolver(T learningRate, T momentum) : learningRate_(learningRate), momentum_(momentum) {
    }

    virtual void solve(Tensor<T>& values, Tensor<T>& grad) {
        size_t size = values.size();
        ASSERT_TRUE(size == grad.size());

        if (!init) {
            prev.resize(size_t(size), size_t(1));
            std::fill_n(prev.data(), size, 0.0f);
            init = true;
        }
        ASSERT_TRUE(prev.size() == size);

        const T* g = grad.data();
        T* w = values.data();
        T* p = prev.data();

        for(size_t j = 0; j < size; ++j) {
            p[j] = momentum_ * p[j] + learningRate_ * g[j];
            w[j] += p[j];
        }
    }

    virtual OptimizationsSolver<T>* clone() {
        return new SGDMomentumSolver<T>(learningRate_, momentum_);
    }

protected:
    T learningRate_;
    T momentum_;
    Tensor<T> prev;
    bool init = false;
};



template<typename InType, typename OutType>
class NNTopology {
public:

    LayerBase<InType, OutType>* add(LayerBase<InType, OutType>* layer) {
        layers.push_back(layer);
        return layer;
    }

    LayerBase<InType, OutType>* add(LayerBase<InType, OutType>* layer,  LayerBase<InType, OutType>* prev) {
        layer->prev = prev;
        prev->next = layer;
        layers.push_back(layer);
        // if (last_layer) {
        //     last_layers.push_back(layer);
        // }
        return layer;
    }

    void fit(Tensor<InType>& data, Tensor<InType>& grth, size_t nEpoch, size_t batchSize, OptimizationsSolver<InType>& solver) {
        auto [d1, d2] = data.getFrameSize();
        auto [gd1, gd2] = grth.getFrameSize();
        size_t xFrameSize = d1 * d2;
        size_t yFrameSize = gd1 * gd2;
        size_t nSamples = data.getNBatches();

        size_t nIter = nSamples / batchSize + !!(nSamples % batchSize);

        std::vector<OptimizationsSolver<InType>*> solvers;

        for(size_t i = 0; i < layers.size(); ++i) {
            solvers.push_back(solver.clone());
        }

        for(size_t iEpoch = 0; iEpoch < nEpoch; iEpoch++) {
            double t1 = omp_get_wtime();
            for(size_t iter = 0; iter < nIter; ++iter) {

                const size_t iStart = iter * batchSize;
                const size_t iEnd =  (iter+1) * batchSize > nSamples ? nSamples : (iter+1) * batchSize;

                Tensor<InType> x(iEnd-iStart, d1, d2);
                Tensor<InType> y(iEnd-iStart, gd1, gd2);

                std::copy_n(data.data() + iStart*xFrameSize, (iEnd-iStart)*xFrameSize, x.data());
                std::copy_n(grth.data() + iStart*yFrameSize, (iEnd-iStart)*yFrameSize, y.data());
                layers[layers.size() - 1]->setGroundTruth(y);

                // forward propogation
                layers[0]->forward(x);
                for(size_t i = 1; i < layers.size(); ++i) {
                    layers[i]->forward(layers[i]->prev->getOutput());
                }

                // backward propogation
                layers[layers.size() - 1]->backward();
                for(int i = int(layers.size()) - 2; i >= 0; --i) {
                    layers[i]->backward(layers[i]->next->getGradient(), i!=0);
                }

                // Update weights with optimization solver
                for(size_t i = 0; i < layers.size(); ++i) {
                    solvers[i]->solve(layers[i]->getWeights(), layers[i]->getWeightsDerivatives());
                }
            }
            double t2 = omp_get_wtime();
            // auto re = layers[layers.size() - 1]->getOutput();
            auto re = predict(data);
            double count = 0;

            InType* gr = grth.data();
            InType* r  = re.data();

            #pragma omp parallel for reduction(+:count)
            for(size_t i = 0; i < nSamples*yFrameSize; ++i) {
                // printf("%f %f\n", grth.data()[i], re.data()[i]);
                count += (gr[i] - r[i]) * (gr[i] - r[i]);
            }

            size_t countAcc = 0;
            for(size_t i = 0; i < nSamples; ++i) {
                auto iter1 = std::max_element(re.data() + yFrameSize*i, re.data() + yFrameSize*i + yFrameSize);
                auto iter2 = std::max_element(grth.data() + yFrameSize*i, grth.data() + yFrameSize*i + yFrameSize);

                // printf("%f %f %d %d\n", *iter1, *iter2, (iter1-(re.data() + yFrameSize*i)), (iter2-(grth.data() + yFrameSize*i)));

                if ((iter1-(re.data() + yFrameSize*i)) == (iter2-(grth.data() + yFrameSize*i))) {
                    countAcc++;
                }
            }
            double t3 = omp_get_wtime();
            printf("[%4zu][%7.3f][%7.3f] RMSE = %f | Acc = %f\n", iEpoch, (t2-t1)*1000, (t3-t2)*1000, std::sqrt(double(count)/double(nSamples)), double(countAcc)/double(nSamples));
        }
    }

    Tensor<InType>& predict(Tensor<InType>& data) {
        layers[0]->forward(data);
        for(size_t i = 1; i < layers.size(); ++i) {
            layers[i]->forward(layers[i]->prev->getOutput());
        }
        return layers[layers.size()-1]->getOutput();
    }

protected:
    std::vector<LayerBase<InType, OutType>*> layers;
    // std::vector<LayerBase<InType, OutType>*> last_layers;
};


#endif // #ifndef __NN_TOPOLOGY_H__
