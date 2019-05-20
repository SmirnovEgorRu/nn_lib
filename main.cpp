
#include <random>
#include <algorithm>
#include "nn.h"


template<typename T>
void generate_simple_func(T* X, T* y, size_t n, size_t p, T dev) {
    std::mt19937 gen{777};
    std::normal_distribution<> d{0, dev};

    for(size_t i = 0; i < n; ++i) {
        T sum = 0;
        for(size_t j = 0; j < p; ++j) {
            sum += X[i * p + j];
        }
        y[i] = sum + d(gen);

        // if (sum > 1) {
        //     y[i] = 1;
        // } else if (sum < 1) {
        //     y[i] = 0;
        // } else {
        //     y[i] = sum;
        // }
    }
}

void test_fullyconnected() {
    size_t nInputNeurons = 3;
    size_t nOutputNeurons = 4;
    size_t batchSize = 1;

    Tensor<float> data(batchSize, nInputNeurons, 1);
    Tensor<float> grad_in(batchSize, nOutputNeurons, 1);

    rand_f<float>(data.data(), data.getDims(), -1.0, 1.0, 111);
    rand_f<float>(grad_in.data(), grad_in.getDims(), -1, 1, 112);

    Fullyconnected<float, float> fc(nInputNeurons, nOutputNeurons);
    fc.forward(data);

    printArr(data.data(), batchSize, nInputNeurons, "x");
    printArr(fc.getWeights().data(), nOutputNeurons, nInputNeurons, "w");
    printArr(fc.getOutput().data(), batchSize, nOutputNeurons, "z");

    fc.backward(grad_in, true);

    printArr(grad_in.data(), batchSize, nOutputNeurons, "grad_in");
    printArr(fc.getGradient().data(), batchSize, nInputNeurons, "grad");
    printArr(fc.getWeightsDerivatives().data(), nOutputNeurons, nInputNeurons, "weights_der");
}

void nn_train(size_t batchSize) {
    size_t n = 10000;
    size_t p = 5;
    size_t nEpoch = 5;
    std::array<size_t, 3> dims = { p, 10, 1 };

    Tensor<float> x(n, p, size_t(1));
    Tensor<float> y(n, size_t(1), size_t(1));

    rand_f<float>(x.data(), n*p, 0, 1, 333);
    generate_simple_func(x.data(), y.data(), n, p, 0.1f);

    NNTopology<float, float> topo;

    auto fc1 = topo.add(new Fullyconnected<float, float>(dims[0], dims[2]));

    SGDSolver<float> solver(0.1);
    topo.fit(x, y, nEpoch, batchSize, solver);
    Tensor<float>& pred = topo.predict(x);
}

void nn_train_2(size_t batchSize) {
    size_t n = 10000;
    size_t p = 5;
    size_t nEpoch = 20;
    std::array<size_t, 3> dims = { p, 10, 1 };

    Tensor<float> x(n, p, size_t(1));
    Tensor<float> y(n, size_t(1), size_t(1));

    rand_f<float>(x.data(), n*p, 0, 1, 333);
    generate_simple_func(x.data(), y.data(), n, p, 0.1f);

    NNTopology<float, float> topo;

    auto fc1 = topo.add(new Fullyconnected<float, float>(dims[0], dims[1]));
    auto sigm = topo.add(new Sigmoid<float, float>(), fc1);
    auto fc2 = topo.add(new Fullyconnected<float, float>(dims[1], dims[2]), sigm);

    SGDSolver<float> solver(0.1);
    topo.fit(x, y, nEpoch, batchSize, solver);
    Tensor<float>& pred = topo.predict(x);
}

void test_convolution() {
    size_t d1 = 3;
    size_t d2 = 3;
    size_t k1 = 2;
    size_t k2 = 2;

    size_t out1 = d1 - k1 + 1;
    size_t out2 = d2 - k2 + 1;
    size_t batchSize = 1;

    Tensor<float> data(batchSize, d1, d2);
    Tensor<float> grad_in(batchSize, out1, out2);

    rand_f<float>(data.data(), data.getDims(), -1.0, 1.0, 111);
    rand_f<float>(grad_in.data(), grad_in.getDims(), -1, 1, 112);

    Convolution2d<float, float> conv({d1, d2, k1, k2});
    conv.forward(data);

    printArr(data.data(), d1, d2, "x");
    printArr(conv.getWeights().data(), k1, k2, "w");
    printArr(conv.getOutput().data(), out1, out2, "z");

    conv.backward(grad_in, true);

    printArr(grad_in.data(), out1, out2, "grad_in");
    printArr(conv.getGradient().data(), d1, d2, "grad");
    printArr(conv.getWeightsDerivatives().data(), k1, k2, "weights_der");
}

void mnist_nn(size_t batchSize) {
    std::string trainDataSet("/nfs/site/proj/mkl/mirror/NN/DAAL_datasets/decision_forest/mnist_train.csv");

    size_t nClasses = 10;
    size_t nEpoch = 5;
    size_t p = 784;
    size_t d = 28;
    float* data;
    float* labels;
    size_t nRows = fast_data_source::read_csv(&data, &labels, p, trainDataSet.c_str());

    printf("nRows = %zu\n", nRows);

    Tensor<float> x(nRows, d, d);
    Tensor<float> y(nRows, nClasses, 1);
    std::copy_n(data, nRows * p, x.data());

    std::fill_n(y.data(), nRows * nClasses, 0.0f);
    for(size_t i = 0; i < nRows; ++i) {
        y.data()[i*nClasses + int(labels[i])] = 1.0f;
    }

    std::array<size_t, 3> dims = { p, 200, nClasses };
    NNTopology<float, float> topo;
    auto fc1 = topo.add(new Fullyconnected<float, float>(dims[0], dims[1]));
    auto sigm1 = topo.add(new Sigmoid<float, float>(), fc1);
    auto fc2 = topo.add(new Fullyconnected<float, float>(dims[1], dims[2]), sigm1);
    auto sigm2 = topo.add(new Sigmoid<float, float>(), fc2);


    SGDSolver<float> solver(0.3);
    topo.fit(x, y, nEpoch, batchSize, solver);
    Tensor<float>& pred = topo.predict(x);
}

template<typename T>
void mnist_nn_conv(size_t batchSize, size_t nEpoch) {
    std::string trainDataSet("/nfs/site/proj/mkl/mirror/NN/DAAL_datasets/decision_forest/mnist_train.csv");

    size_t nClasses = 10;
    size_t p = 784;
    size_t d = 28;
    float* data;
    float* labels;
    size_t nRows = fast_data_source::read_csv(&data, &labels, p, trainDataSet.c_str());

    printf("nRows = %zu\n", nRows);

    Tensor<T> x(nRows, d, d);
    Tensor<T> y(nRows, nClasses, 1);
    std::copy_n(data, nRows * p, x.data());

    std::fill_n(y.data(), nRows * nClasses, 0.0f);
    for(size_t i = 0; i < nRows; ++i) {
        y.data()[i*nClasses + int(labels[i])] = 1.0f;
    }

    std::array<size_t, 3> dims = { p, 200, nClasses };
    NNTopology<T, T> topo;

    auto conv1 = topo.add(new Convolution2d<T, T>({28, 28, 5, 5}));
    auto sigm1 = topo.add(new Sigmoid<T, T>(), conv1);

    auto conv2 = topo.add(new Convolution2d<T, T>({24, 24, 5, 5}), sigm1);
    auto sigm2 = topo.add(new Sigmoid<T, T>(), conv2);

    auto conv3 = topo.add(new Convolution2d<T, T>({20, 20, 5, 5}), sigm2);
    auto sigm3 = topo.add(new Sigmoid<T, T>(), conv3);

    auto fc1 = topo.add(new Fullyconnected<T, T>(16*16, 100), sigm3);
    auto sigm4 = topo.add(new Sigmoid<T, T>(), fc1);

    auto fc2 = topo.add(new Fullyconnected<T, T>(100, nClasses), sigm4);
    auto sigm5 = topo.add(new Sigmoid<T, T>(), fc2);

    SGDMomentumSolver<T> solver(0.3, 0.8);
    topo.fit(x, y, nEpoch, batchSize, solver);
    Tensor<T>& pred = topo.predict(x);
}

int main(int argc, char* argv[]) {
    size_t batchSize = 1;
    size_t nEpoch = 5;
    if (argc > 1)
    {
        batchSize = std::atoi(argv[1]);
    }
    if (argc > 2)
    {
        nEpoch = std::atoi(argv[2]);
    }
    // test_convolution();
    mnist_nn_conv<float>(batchSize, nEpoch);
    // mnist_nn_conv<double>(batchSize, nEpoch);

    // test_fullyconnected();
    // nn_train(batchSize);
    // printf("\n");
    // nn_train_2(batchSize);
    // mnist_nn(batchSize);

    return 0;
}
