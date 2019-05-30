
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

void test_convolution_vec() {
    size_t batchSize = 1;

    std::vector<size_t> d1_vec({5, 16, 17, 28, 32, 37, 64, 77, 128});
    std::vector<size_t> d2_vec({5, 16, 17, 28, 32, 37, 64, 77, 128});
    std::vector<size_t> k1_vec({2, 3, 4, 5, 7});
    std::vector<size_t> k2_vec({2, 3, 4, 5, 7});

    for(size_t ipk = 0; ipk < k1_vec.size(); ++ipk) {
        for(size_t ip = 0; ip < d1_vec.size(); ++ip) {
            size_t d1 = d1_vec[ip];
            size_t d2 = d2_vec[ip];
            size_t k1 = k1_vec[ipk];
            size_t k2 = k2_vec[ipk];

            if(k1 >= d1) continue;

            size_t out1 = d1 - k1 + 1;
            size_t out2 = d2 - k2 + 1;

            std::vector<float> data(d1 * d2);
            std::vector<float> kernel(k1 * k2);
            std::vector<float> res1(out1 * out2);
            std::vector<float> res2(out1 * out2);

            rand_f<float>(data.data(), data.size(), -1.0, 1.0, 111);
            rand_f<float>(kernel.data(), kernel.size(), -1.0, 1.0, 111);

            std::fill(res1.begin(), res1.end(), 0.0f);
            std::fill(res2.begin(), res2.end(), 0.0f);

            std::vector<size_t> cc;
            size_t N_REPEAT = 1000;
            for(size_t i = 0; i < N_REPEAT; ++i) {
                size_t c1 = _rdtsc();
                conv_simple(data.data(), kernel.data(), res1.data(), d1, d2, k1, k2);
                size_t c2 = _rdtsc();
                cc.push_back(c2-c1);
            }
            std::sort(cc.begin(), cc.begin());
            size_t c1 = cc[N_REPEAT/2];
            // size_t c1 = cc[2];

            cc.clear();
            for(size_t i = 0; i < N_REPEAT; ++i) {
                size_t c1 = _rdtsc();
                _conv_avx512(data.data(), kernel.data(), res2.data(), d1, d2, k1, k2);
                size_t c2 = _rdtsc();
                cc.push_back(c2-c1);
            }
            std::sort(cc.begin(), cc.begin());
            size_t c2 = cc[N_REPEAT/2];
            // size_t c2 = cc[2];

            // size_t c1 = _rdtsc();
            // _conv_simple(data.data(), kernel.data(), res1.data(), d1, d2, k1, k2);
            // size_t c2 = _rdtsc();
            // _conv_avx512(data.data(), kernel.data(), res2.data(), d1, d2, k1, k2);
            // size_t c3 = _rdtsc();

            // printf("CPE: %zu %zu\n", c2-c1, c3-c2);
            float cpe1 = float(c1)/(k1*k2*out1*out2);
            float cpe2 = float(c2)/(k1*k2*out1*out2);


            printf("CPE: %4zu x %4zu, %4zu x %4zu | %7.3f %7.3f | %7.3f\n", d1, d2, k1, k2, cpe1, cpe2, cpe1 / cpe2);

            bool equal = std::equal(res1.begin(), res1.end(), res2.begin(), [](float r1, float r2) {
                return std::isfinite(r1) && std::isfinite(r2) && std::abs(r1 - r2) < 0.001;
            });
            if (!equal) {
                printf("FAILED!\n");

                printArr(data.data(), d1, d2, "x");
                printArr(kernel.data(), k1, k2, "w");

                printArr(res1.data(), out1, out2, "ref");
                printArr(res2.data(), out1, out2, "avx");
                return;
            }
        }
    }
    printf("PASSED!\n");
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
    // auto sm5 = topo.add(new SoftMaxLayer<T, T>(), fc2);


    // auto conv1 = topo.add(new Convolution2d<T, T>({28, 28, 5, 5}));
    // auto sigm1 = topo.add(new SoftMaxLayer<T, T>(), conv1);

    // auto conv2 = topo.add(new Convolution2d<T, T>({24, 24, 5, 5}), sigm1);
    // auto sigm2 = topo.add(new SoftMaxLayer<T, T>(), conv2);

    // auto conv3 = topo.add(new Convolution2d<T, T>({20, 20, 5, 5}), sigm2);
    // auto sigm3 = topo.add(new SoftMaxLayer<T, T>(), conv3);

    // auto fc1 = topo.add(new Fullyconnected<T, T>(16*16, 100), sigm3);
    // auto sigm4 = topo.add(new SoftMaxLayer<T, T>(), fc1);

    // auto fc2 = topo.add(new Fullyconnected<T, T>(100, nClasses), sigm4);
    // auto sm5 = topo.add(new SoftMaxLayer<T, T>(), fc2);


    SGDMomentumSolver<T> solver(0.3, 0.5);
    topo.fit(x, y, nEpoch, batchSize, solver);
    // Tensor<T>& pred = topo.predict(x);
    topo.score(x, y);
    COMPUTE_IN_INT8 = true;
    topo.score(x, y);
}

void test_dpbusd() {
    // const __m512i x = _mm512_set1_epi16(200);
    // const __m512i k = _mm512_set1_epi16(-3);
    // const __m512i src = _mm512_set1_epi32(1);

    std::vector<uint8_t> data(64);
    std::vector<int8_t> kernel(64);
    std::vector<int32_t> res1(16);

    for(size_t i = 0; i < data.size(); ++i) {
        data[i] = i+1;
    }
    for(size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = kernel.size() - i + 1;
    }

    const __m512i x = _mm512_loadu_epi8(data.data());
    const __m512i k = _mm512_loadu_epi8(kernel.data());
    const __m512i src = _mm512_loadu_epi32(res1.data());

    int32_t arr[16];

    __m512i res = dpbusd(x, k , src);
    _mm512_storeu_epi32(arr, res);

    for(size_t i = 0; i < 16; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

template<typename Func>
size_t measure(Func func, size_t N_REPEAT = 1000) {
    std::vector<size_t> cc;
    for(size_t i = 0; i < N_REPEAT; ++i) {
        size_t c1 = _rdtsc();
        func();
        size_t c2 = _rdtsc();
        cc.push_back(c2-c1);
    }
    std::sort(cc.begin(), cc.begin());
    size_t c1 = cc[N_REPEAT/2];
    return c1;
}


double conv_bench_int8_vnni(int d1, int d2, int k1, int k2, int out1, int out2) {
    std::vector<uint8_t> data(d1 * d2);
    std::vector<int8_t> kernel(k1 * k2);
    std::vector<int32_t> res1(out1 * out2);
    for(size_t i = 0; i < data.size(); ++i) {
        data[i] = i+1;
    }
    for(size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = kernel.size() - i + 1;
    }

    size_t clocks = measure([&]() {
        vnni_conv(data.data(), kernel.data(), res1.data(), d1, d2, k1, k2);
    }, 50000);

    double cpe = double(clocks)/(k1*k2*out1*out2);
    return cpe;
}

double conv_bench_int8_native(int d1, int d2, int k1, int k2, int out1, int out2) {
    std::vector<uint8_t> data(d1 * d2);
    std::vector<int8_t> kernel(k1 * k2);
    std::vector<int32_t> res1(out1 * out2);
    for(size_t i = 0; i < data.size(); ++i) {
        data[i] = i+1;
    }
    for(size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = kernel.size() - i + 1;
    }

    size_t clocks = measure([&]() {
        _conv_simple<uint8_t, int16_t, int8_t, int16_t, int32_t>(data.data(), kernel.data(), res1.data(), d1, d2, k1, k2);
    });

    double cpe = double(clocks)/(k1*k2*out1*out2);
    return cpe;
}

double conv_bench_float_native(int d1, int d2, int k1, int k2, int out1, int out2) {
    std::vector<float> data(d1 * d2);
    std::vector<float> kernel(k1 * k2);
    std::vector<float> res1(out1 * out2);
    for(size_t i = 0; i < data.size(); ++i) {
        data[i] = i+1;
    }
    for(size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = kernel.size() - i + 1;
    }

    size_t clocks = measure([&]() {
        conv_simple(data.data(), kernel.data(), res1.data(), d1, d2, k1, k2);
    });

    double cpe = double(clocks)/(k1*k2*out1*out2);
    return cpe;
}

double conv_bench_float_avx512(int d1, int d2, int k1, int k2, int out1, int out2) {
    std::vector<float> data(d1 * d2);
    std::vector<float> kernel(k1 * k2);
    std::vector<float> res1(out1 * out2);
    for(size_t i = 0; i < data.size(); ++i) {
        data[i] = i+1;
    }
    for(size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = kernel.size() - i + 1;
    }

    size_t clocks = measure([&]() {
        _conv_avx512(data.data(), kernel.data(), res1.data(), d1, d2, k1, k2);
    });

    double cpe = double(clocks)/(k1*k2*out1*out2);
    return cpe;
}


void running_conv_bench() {
    std::vector<size_t> d1_vec({5, 16, 17, 28, 32, 37, 64, 77, 128});
    std::vector<size_t> d2_vec({5, 16, 17, 28, 32, 37, 64, 77, 128});
    std::vector<size_t> k1_vec({2, 3, 4, 5, 7});
    std::vector<size_t> k2_vec({2, 3, 4, 5, 7});

    for(size_t ipk = 0; ipk < k1_vec.size(); ++ipk) {
        for(size_t ip = 0; ip < d1_vec.size(); ++ip) {
            size_t d1 = d1_vec[ip];
            size_t d2 = d2_vec[ip];
            size_t k1 = k1_vec[ipk];
            size_t k2 = k2_vec[ipk];

            if(k1 > d1) continue;
            if(k2 > d2) continue;

            size_t out1 = d1 - k1 + 1;
            size_t out2 = d2 - k2 + 1;

            // double t1 = conv_bench_float_native(d1, d2, k1, k2, out1, out2);
            // double t2 = conv_bench_float_avx512(d1, d2, k1, k2, out1, out2);
            // double t3 = conv_bench_int8_native(d1, d2, k1, k2, out1, out2);
            // double t4 = conv_bench_int8_vnni(d1, d2, k1, k2, out1, out2);


            double t1 = 1;
            double t2 = 1;
            double t3 = 1;
            double t4 = conv_bench_int8_vnni(d1, d2, k1, k2, out1, out2);

            printf("CPE: %4zu x %4zu, %4zu x %4zu | %7.3f %7.3f | %7.3f %7.3f | %7.3f %7.3f %7.3f\n", d1, d2, k1, k2, t1, t2, t3, t4, t1/t2, t1/t3, t1/t4);
        }
    }


}


void test_convolution_vnni() {
    printf("test_convolution_vnni\n");
    size_t batchSize = 1;

    std::vector<size_t> d1_vec({5, 16, 17, 28, 32, 37, 64, 77, 128});
    std::vector<size_t> d2_vec({5, 16, 17, 28, 32, 37, 64, 77, 128});
    std::vector<size_t> k1_vec({2, 3, 4, 5, 7});
    std::vector<size_t> k2_vec({2, 3, 4, 5, 7});

    size_t n_executed_tests = 0;

    for(size_t ipk = 0; ipk < k1_vec.size(); ++ipk) {
        for(size_t ip = 0; ip < d1_vec.size(); ++ip) {
            size_t d1 = d1_vec[ip];
            size_t d2 = d2_vec[ip];
            size_t k1 = k1_vec[ipk];
            size_t k2 = k2_vec[ipk];

            if(k1 > d1) continue;
            if(k2 > d2) continue;

            size_t out1 = d1 - k1 + 1;
            size_t out2 = d2 - k2 + 1;

            std::vector<uint8_t> data(d1 * d2);
            std::vector<int8_t> kernel(k1 * k2);
            std::vector<int32_t> res1(out1 * out2);
            std::vector<int32_t> res2(out1 * out2);

            // rand_f<float>(data.data(), data.size(), -1.0, 1.0, 111);
            // rand_f<float>(kernel.data(), kernel.size(), -1.0, 1.0, 111);


            for(size_t i = 0; i < data.size(); ++i) {
                data[i] = i+1;
            }
            for(size_t i = 0; i < kernel.size(); ++i) {
                kernel[i] = kernel.size() - i + 1;
            }


            // std::fill(data.begin(), data.end(), 1);
            // std::fill(kernel.begin(), kernel.end(), 2);

            // std::fill(res1.begin(), res1.end(), 0.0f);
            // std::fill(res2.begin(), res2.end(), 0.0f);

            std::vector<size_t> cc;
            size_t N_REPEAT = 1000;
            for(size_t i = 0; i < N_REPEAT; ++i) {
                size_t c1 = _rdtsc();
                _conv_simple<uint8_t, int16_t, int8_t, int16_t, int32_t>(data.data(), kernel.data(), res1.data(), d1, d2, k1, k2);
                size_t c2 = _rdtsc();
                cc.push_back(c2-c1);
            }
            std::sort(cc.begin(), cc.begin());
            size_t c1 = cc[N_REPEAT/2];
            // size_t c1 = cc[2];

            cc.clear();
            for(size_t i = 0; i < N_REPEAT; ++i) {
                size_t c1 = _rdtsc();
                // _conv_avx512(data.data(), kernel.data(), res2.data(), d1, d2, k1, k2);
                vnni_conv(data.data(), kernel.data(), res2.data(), d1, d2, k1, k2);
                size_t c2 = _rdtsc();
                cc.push_back(c2-c1);
            }
            std::sort(cc.begin(), cc.begin());
            size_t c2 = cc[N_REPEAT/2];
            // size_t c2 = cc[2];

            // size_t c1 = _rdtsc();
            // _conv_simple(data.data(), kernel.data(), res1.data(), d1, d2, k1, k2);
            // size_t c2 = _rdtsc();
            // _conv_avx512(data.data(), kernel.data(), res2.data(), d1, d2, k1, k2);
            // size_t c3 = _rdtsc();

            // printf("CPE: %zu %zu\n", c2-c1, c3-c2);
            float cpe1 = float(c1)/(k1*k2*out1*out2);
            float cpe2 = float(c2)/(k1*k2*out1*out2);

            printf("CPE: %4zu x %4zu, %4zu x %4zu | %7.3f %7.3f | %7.3f\n", d1, d2, k1, k2, cpe1, cpe2, cpe1 / cpe2);

            bool equal = std::equal(res1.begin(), res1.end(), res2.begin(), [](float r1, float r2) {
                return std::isfinite(r1) && std::isfinite(r2) && std::abs(r1 - r2) < 0.001;
            });
            if (!equal) {
                n_executed_tests++;
                printf("FAILED!\n");

                // printArr(data.data(), d1, d2, "x");
                // printArr(kernel.data(), k1, k2, "w");

                // printArr(res1.data(), out1, out2, "ref");
                // printArr(res2.data(), out1, out2, "avx");
                // return;
            } else {
                n_executed_tests++;
                // printArr(res1.data(), out1, out2, "ref");
                // printArr(res2.data(), out1, out2, "avx");
            }
        }
    }

    if (!n_executed_tests) {
        printf("SKIPPED\n");
    } else {
        printf("PASSED! \n");
    }
}

void test_my_mm512_permutexvar_epi8() {

    const __m512i idx = _mm512_set_epi8(    18, 17, 16, 15,
    17, 16, 15, 14,
    16, 15, 14, 13,
    15, 14, 13, 12,
    14, 13, 12, 11,
    13, 12, 11, 10,
    12, 11, 10,  9,
    11, 10,  9,  8,
    10,  9,  8,  7,
     9,  8,  7,  6,
     8,  7,  6,  5,
     7, 6, 5,4,
     6,5,4,3,
     5,4,3,2,
     4,3,2,1,
     3,2,1,0);


    uint8_t arr[64];
    for(size_t i = 0; i < 64; ++i) {
        arr[i] = i + 1;
    }
    const __m512i x = _mm512_loadu_epi8(arr);

    __m512i res = permutexvar_epi8(idx, x);

    _mm512_storeu_epi8(arr, res);

    for(size_t i = 0; i < 64; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
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
    // mnist_nn_conv<float>(batchSize, nEpoch);

    running_conv_bench();

    // test_dpbusd();
    // test_convolution_vec();
    // test_convolution_vnni();
    // test_my_mm512_permutexvar_epi8();


    // test_convolution_vec();
    // mnist_nn_conv<double>(batchSize, nEpoch);

    // test_fullyconnected();
    // nn_train(batchSize);
    // printf("\n");
    // nn_train_2(batchSize);
    // mnist_nn(batchSize);

    return 0;
}
