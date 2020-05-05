
#include <stdio.h>
#include <string>
#include "nn.h"
#undef ASSERT_TRUE
#include <gtest/gtest.h>


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


size_t read_data_set(std::string path, Tensor<float>& x, Tensor<float>& y, size_t p, size_t nClasses) {
    float* data;
    float* labels;
    size_t nRows = fast_data_source::read_csv(&data, &labels, p, path.c_str());

    x.resize(nRows, std::sqrt(p), std::sqrt(p));
    y.resize(nRows, nClasses, 1);
    std::copy_n(data, nRows * p, x.data());

    std::fill_n(y.data(), nRows * nClasses, 0.0f);
    for(size_t i = 0; i < nRows; ++i) {
        y.data()[i*nClasses + int(labels[i])] = 1.0f;
    }

    return nRows;
}

TEST(mnist, train_fullyconnected_nn)
{
    std::string datasetPath = "../data";
    std::string trainDataSet(datasetPath + "/train.csv");
    std::string testDataSet(datasetPath + "/test.csv");

    size_t nClasses = 10;
    size_t batchSize = 256;
    size_t nEpoch = 5;
    size_t p = 784;
    size_t d = 28;

    Tensor<float> x, y;
    size_t nRows = read_data_set(trainDataSet, x, y, p, nClasses);

    std::array<size_t, 3> dims = { p, 1000, nClasses };
    NNTopology<float, float> topo;
    auto fc1 = topo.add(new Fullyconnected<float, float>(dims[0], dims[1]));
    auto sigm1 = topo.add(new Sigmoid<float, float>(), fc1);
    auto fc2 = topo.add(new Fullyconnected<float, float>(dims[1], dims[2]), sigm1);
    auto sigm2 = topo.add(new SoftMaxLayer<float, float>(), fc2);

    SGDSolver<float> solver(0.3);

    double t1 = omp_get_wtime();
    topo.fit(x, y, nEpoch, batchSize, solver);
    double t2 = omp_get_wtime();

    printf("Fit time = %f sec\n", t2-t1);

    Tensor<float> xTest, yTest;
    size_t nRowsTest = read_data_set(testDataSet, xTest, yTest, p, nClasses);

    Tensor<float>& pred = topo.predict(xTest);

    size_t countAcc = 0;
    #pragma omp parallel for reduction(+:countAcc)
    for(size_t i = 0; i < nRowsTest; ++i) {
        auto iter1 = std::max_element(pred.data() + nClasses*i, pred.data() + nClasses*i + nClasses);
        auto iter2 = std::max_element(yTest.data() + nClasses*i, yTest.data() + nClasses*i + nClasses);

        if ((iter1-(pred.data() + nClasses*i)) == (iter2-(yTest.data() + nClasses*i))) {
            countAcc++;
        }
    }
    auto acc = 100.f * float(countAcc)/nRowsTest;

    printf("Test accuracy = %f\n", acc);
    ASSERT_GT(acc, 99.7);
}

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
