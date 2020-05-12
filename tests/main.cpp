
#include <stdio.h>
#include <string>
#include "nn.h"
#undef ASSERT_TRUE
#include <gtest/gtest.h>
#include "tbb/scalable_allocator.h"
#include "tbb/cache_aligned_allocator.h"

static uint64_t _rdtsc(void) {
    uint64_t rax, rdx;
    __asm__ __volatile__("rdtsc" : "=a"(rax), "=d"(rdx) : :);
    return (rdx << 32) | rax;
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

TEST(conv, VNNI)
{
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
            size_t N_REPEAT = 5;
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

            float TOPS = out1 * out2 * 2 * k1 * k2;
            float TOPS_PER_CLOCK = 128;

            bool equal = std::equal(res1.begin(), res1.end(), res2.begin(), [](float r1, float r2) {
                return std::isfinite(r1) && std::isfinite(r2) && std::abs(r1 - r2) < 0.001;
            });


            if (!equal) {

                printf("FAILED!: %4zu x %4zu, %4zu x %4zu | %7.3f %7.3f | %7.3f\n", d1, d2, k1, k2, cpe1, cpe2, cpe1 / cpe2);

                // printArr(data.data(), d1, d2, "x");
                // printArr(kernel.data(), k1, k2, "w");

                printArr(res1.data(), out1, out2, "ref");
                printArr(res2.data(), out1, out2, "avx");
                return;
            } else {
                // printArr(res1.data(), out1, out2, "ref");
                // printArr(res2.data(), out1, out2, "avx");
            }
            ASSERT_TRUE(equal);
            printf("%4zu x %4zu, %4zu x %4zu | %7.3f %7.3f | %7.3f | %7.3f vs. peak %7.3f  = %7.3f \n", d1, d2, k1, k2, cpe1, cpe2, cpe1 / cpe2,  TOPS/c2, TOPS_PER_CLOCK, (TOPS/c2)/TOPS_PER_CLOCK*100);
        }
    }
}


TEST(conv, VNNI_mnist)
{
    size_t batchSize = 1;

    std::vector<size_t> d1_vec({28});
    std::vector<size_t> d2_vec({28});
    std::vector<size_t> k1_vec({5});
    std::vector<size_t> k2_vec({5});

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
            size_t N_REPEAT = 5;
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

            float TOPS = out1 * out2 * 2 * k1 * k2;
            float TOPS_PER_CLOCK = 128;


            bool equal = std::equal(res1.begin(), res1.end(), res2.begin(), [](float r1, float r2) {
                return std::isfinite(r1) && std::isfinite(r2) && std::abs(r1 - r2) < 0.001;
            });


            if (!equal) {

                printf("FAILED!: %4zu x %4zu, %4zu x %4zu | %7.3f %7.3f | %7.3f\n", d1, d2, k1, k2, cpe1, cpe2, cpe1 / cpe2);

                // printArr(data.data(), d1, d2, "x");
                // printArr(kernel.data(), k1, k2, "w");

                printArr(res1.data(), out1, out2, "ref");
                printArr(res2.data(), out1, out2, "avx");
                return;
            } else {
                // printArr(res1.data(), out1, out2, "ref");
                // printArr(res2.data(), out1, out2, "avx");
            }
            ASSERT_TRUE(equal);
            printf("%4zu x %4zu, %4zu x %4zu | %7.3f %7.3f | %7.3f | %7.3f vs. peak %7.3f  = %7.3f \n", d1, d2, k1, k2, cpe1, cpe2, cpe1 / cpe2, TOPS/c2, TOPS_PER_CLOCK, (TOPS/c2)/TOPS_PER_CLOCK*100);
        }
    }
}


TEST(conv, VNNI_reorder)
{
    size_t batchSize = 1;

    std::vector<size_t> d1_vec({16, 17, 28, 32, 37, 64, 77, 128});
    std::vector<size_t> d2_vec({16, 17, 28, 32, 37, 64, 77, 128});
    std::vector<size_t> k1_vec({2, 3, 4, 5, 7, 8});
    std::vector<size_t> k2_vec({2, 3, 4, 5, 7, 8});

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

            // std::vector<uint8_t, tbb::cache_aligned_allocator<uint8_t>> data(d1 * d2);
            // std::vector<int8_t, tbb::cache_aligned_allocator<int8_t>> kernel(k1 * k2);
            // std::vector<int32_t, tbb::cache_aligned_allocator<int32_t>> res1(out1 * out2);
            // std::vector<int32_t, tbb::cache_aligned_allocator<int32_t>> res2(out1 * out2);

            size_t n_kernels = 50;

            size_t align = 128;
            uint8_t* data = (uint8_t*)_mm_malloc(4 * d1 * d2, align);
            int8_t* kernel = (int8_t*)_mm_malloc(4 * k1 * k2 * n_kernels, align);
            int32_t* res1 = (int32_t*)_mm_malloc(4 * out1 * out2 * n_kernels, align);
            int32_t* res2 = (int32_t*)_mm_malloc(4 * out1 * out2 * n_kernels, align);

            // rand_f<float>(data.data(), data.size(), -1.0, 1.0, 111);
            // rand_f<float>(kernel.data(), kernel.size(), -1.0, 1.0, 111);


            for(size_t i = 0; i < d1 * d2; ++i) {
                data[i] = i+1;
            }
            for(size_t i = 0; i < k1 * k2 * n_kernels; ++i) {
                kernel[i] = k1 * k2 - i + 1;
            }


            // std::fill(data.begin(), data.end(), 1);
            // std::fill(kernel.begin(), kernel.end(), 2);

            // std::fill(res1.begin(), res1.end(), 0.0f);
            // std::fill(res2.begin(), res2.end(), 0.0f);

            std::vector<size_t> cc;
            std::vector<double> times;
            size_t N_REPEAT = 15;
            for(size_t i = 0; i < N_REPEAT; ++i) {
                double t1 = omp_get_wtime();
                size_t c1 = _rdtsc();

                // #pragma omp parallel for
                for(size_t ikernel = 0; ikernel < n_kernels; ikernel++)
                {
                    _conv_simple<uint8_t, int16_t, int8_t, int16_t, int32_t>(data, kernel + ikernel * k1 * k2, res1, d1, d2, k1, k2);
                }

                // _conv_simple<uint8_t, int16_t, int8_t, int16_t, int32_t>(data, kernel, res1, d1, d2, k1, k2);
                size_t c2 = _rdtsc();
                double t2 = omp_get_wtime();
                cc.push_back(c2-c1);
                times.push_back(t2-t1);
            }
            std::sort(cc.begin(), cc.end());
            std::sort(times.begin(), times.end());
            size_t c1 = cc[0];
            double t1 = times[0];
            // size_t c1 = cc[2];

            size_t size_buffer = vnni_conv_get_buffer_size(data, kernel, res2, d1, d2, k1, k2);
            // std::vector<uint8_t, tbb::cache_aligned_allocator<uint8_t>> tmp(size_buffer);

            uint8_t* tmp = (uint8_t*)_mm_malloc(4 * size_buffer, align);

            vnni_conv_fill_buffer(data, kernel, res2, d1, d2, k1, k2, tmp);

            cc.clear();
            times.clear();
            for(size_t i = 0; i < N_REPEAT; ++i) {
                double t1 = omp_get_wtime();
                size_t c1 = _rdtsc();
                // _conv_avx512(data.data(), kernel.data(), res2.data(), d1, d2, k1, k2);
                // vnni_conv_buffer(tmp, kernel, res2, d1, d2, k1, k2);

                // #pragma omp parallel for
                for(size_t ikernel = 0; ikernel < n_kernels; ikernel++)
                {
                    vnni_conv_buffer(tmp, kernel + ikernel * k1 * k2, res2, d1, d2, k1, k2);
                }

                size_t c2 = _rdtsc();
                double t2 = omp_get_wtime();
                cc.push_back(c2-c1);
                times.push_back(t2-t1);
            }
            std::sort(cc.begin(), cc.end());
            std::sort(times.begin(), times.end());
            size_t c2 = cc[0];
            double t2 = times[0];
            // size_t c2 = cc[2];

            // size_t c1 = _rdtsc();
            // _conv_simple(data.data(), kernel.data(), res1.data(), d1, d2, k1, k2);
            // size_t c2 = _rdtsc();
            // _conv_avx512(data.data(), kernel.data(), res2.data(), d1, d2, k1, k2);
            // size_t c3 = _rdtsc();

            // printf("CPE: %zu %zu\n", c2-c1, c3-c2);
            float cpe1 = float(c1)/(k1*k2*out1*out2);
            float cpe2 = float(c2)/(k1*k2*out1*out2);

            float TOPS = out1 * out2 * 2 * k1 * k2 * n_kernels;
            float TOPS_PER_CLOCK = 128;


            bool equal = std::equal(res1, res1 + out1*out2, res2, [](float r1, float r2) {
                return std::isfinite(r1) && std::isfinite(r2) && std::abs(r1 - r2) < 0.001;
            });


            if (!equal) {

                printf("FAILED!: %4zu x %4zu, %4zu x %4zu | %7.3f %7.3f | %7.3f\n", d1, d2, k1, k2, cpe1, cpe2, cpe1 / cpe2);

                // printArr(data.data(), d1, d2, "x");
                // printArr(kernel.data(), k1, k2, "w");

                printArr(res1, out1, out2, "ref");
                printArr(res2, out1, out2, "avx");
            } else {
                // printArr(res1.data(), out1, out2, "ref");
                // printArr(res2.data(), out1, out2, "avx");
            }
            ASSERT_TRUE(equal);
            // printf("%4zu x %4zu, %4zu x %4zu | %7.3f %7.3f | %7.3f | %7.3f vs. peak %7.3f  = %7.4f | %7.5f ms | %7.5f ms | buffer = %zu\n", d1, d2, k1, k2, cpe1, cpe2, cpe1 / cpe2, TOPS/c2, TOPS_PER_CLOCK, (TOPS/c2)/TOPS_PER_CLOCK*100, double(c2)/3e6, t2*1000.0, size_buffer/64);


            printf("\"{%zu x %zu} x {%zu x %zu} x %zu\": { \"in1\": %zu, \"in2\": %zu, \"k1\": %zu, \"k2\": %zu, \"n_kernels\": %zu, \"time\": %7.5f / 1000, \"n_cache_lines_x\": %zu, },\n",
                d1, d2, k1, k2, n_kernels, d1, d2, k1, k2, n_kernels, double(c2)/3e6, size_buffer/64);

            _mm_free(data);
            _mm_free(kernel);
            _mm_free(res1);
            _mm_free(res2);
        }
    }
}



TEST(conv, VNNI_old)
{
    size_t batchSize = 1;

    std::vector<size_t> d1_vec({16, 17, 28, 32, 37, 64, 77, 128});
    std::vector<size_t> d2_vec({16, 17, 28, 32, 37, 64, 77, 128});
    std::vector<size_t> k1_vec({2, 3, 4, 5, 7, 8});
    std::vector<size_t> k2_vec({2, 3, 4, 5, 7, 8});

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

            // std::vector<uint8_t, tbb::cache_aligned_allocator<uint8_t>> data(d1 * d2);
            // std::vector<int8_t, tbb::cache_aligned_allocator<int8_t>> kernel(k1 * k2);
            // std::vector<int32_t, tbb::cache_aligned_allocator<int32_t>> res1(out1 * out2);
            // std::vector<int32_t, tbb::cache_aligned_allocator<int32_t>> res2(out1 * out2);

            size_t n_kernels = 50;

            size_t align = 128;
            uint8_t* data = (uint8_t*)_mm_malloc(4 * d1 * d2, align);
            int8_t* kernel = (int8_t*)_mm_malloc(4 * k1 * k2 * n_kernels, align);
            int32_t* res1 = (int32_t*)_mm_malloc(4 * out1 * out2 * n_kernels, align);
            int32_t* res2 = (int32_t*)_mm_malloc(4 * out1 * out2 * n_kernels, align);

            // rand_f<float>(data.data(), data.size(), -1.0, 1.0, 111);
            // rand_f<float>(kernel.data(), kernel.size(), -1.0, 1.0, 111);


            for(size_t i = 0; i < d1 * d2; ++i) {
                data[i] = i+1;
            }
            for(size_t i = 0; i < k1 * k2 * n_kernels; ++i) {
                kernel[i] = k1 * k2 - i + 1;
            }


            // std::fill(data.begin(), data.end(), 1);
            // std::fill(kernel.begin(), kernel.end(), 2);

            // std::fill(res1.begin(), res1.end(), 0.0f);
            // std::fill(res2.begin(), res2.end(), 0.0f);

            std::vector<size_t> cc;
            std::vector<double> times;
            size_t N_REPEAT = 15;
            for(size_t i = 0; i < N_REPEAT; ++i) {
                double t1 = omp_get_wtime();
                size_t c1 = _rdtsc();

                // #pragma omp parallel for
                for(size_t ikernel = 0; ikernel < n_kernels; ikernel++)
                {
                    _conv_simple<uint8_t, int16_t, int8_t, int16_t, int32_t>(data, kernel + ikernel * k1 * k2, res1, d1, d2, k1, k2);
                }

                // _conv_simple<uint8_t, int16_t, int8_t, int16_t, int32_t>(data, kernel, res1, d1, d2, k1, k2);
                size_t c2 = _rdtsc();
                double t2 = omp_get_wtime();
                cc.push_back(c2-c1);
                times.push_back(t2-t1);
            }
            std::sort(cc.begin(), cc.end());
            std::sort(times.begin(), times.end());
            size_t c1 = cc[0];
            double t1 = times[0];
            // size_t c1 = cc[2];

            size_t size_buffer = vnni_conv_get_buffer_size(data, kernel, res2, d1, d2, k1, k2);
            // std::vector<uint8_t, tbb::cache_aligned_allocator<uint8_t>> tmp(size_buffer);

            uint8_t* tmp = (uint8_t*)_mm_malloc(4 * size_buffer, align);

            vnni_conv_fill_buffer(data, kernel, res2, d1, d2, k1, k2, tmp);

            cc.clear();
            times.clear();
            for(size_t i = 0; i < N_REPEAT; ++i) {
                double t1 = omp_get_wtime();
                size_t c1 = _rdtsc();
                // _conv_avx512(data.data(), kernel.data(), res2.data(), d1, d2, k1, k2);
                // vnni_conv_buffer(tmp, kernel, res2, d1, d2, k1, k2);

                // #pragma omp parallel for
                for(size_t ikernel = 0; ikernel < n_kernels; ikernel++)
                {
                    vnni_conv(data, kernel + ikernel * k1 * k2, res2, d1, d2, k1, k2);
                }

                size_t c2 = _rdtsc();
                double t2 = omp_get_wtime();
                cc.push_back(c2-c1);
                times.push_back(t2-t1);
            }
            std::sort(cc.begin(), cc.end());
            std::sort(times.begin(), times.end());
            size_t c2 = cc[0];
            double t2 = times[0];
            // size_t c2 = cc[2];

            // size_t c1 = _rdtsc();
            // _conv_simple(data.data(), kernel.data(), res1.data(), d1, d2, k1, k2);
            // size_t c2 = _rdtsc();
            // _conv_avx512(data.data(), kernel.data(), res2.data(), d1, d2, k1, k2);
            // size_t c3 = _rdtsc();

            // printf("CPE: %zu %zu\n", c2-c1, c3-c2);
            float cpe1 = float(c1)/(k1*k2*out1*out2);
            float cpe2 = float(c2)/(k1*k2*out1*out2);

            float TOPS = out1 * out2 * 2 * k1 * k2 * n_kernels;
            float TOPS_PER_CLOCK = 128;


            bool equal = std::equal(res1, res1 + out1*out2, res2, [](float r1, float r2) {
                return std::isfinite(r1) && std::isfinite(r2) && std::abs(r1 - r2) < 0.001;
            });


            if (!equal) {

                printf("FAILED!: %4zu x %4zu, %4zu x %4zu | %7.3f %7.3f | %7.3f\n", d1, d2, k1, k2, cpe1, cpe2, cpe1 / cpe2);

                // printArr(data.data(), d1, d2, "x");
                // printArr(kernel.data(), k1, k2, "w");

                printArr(res1, out1, out2, "ref");
                printArr(res2, out1, out2, "avx");
            } else {
                // printArr(res1.data(), out1, out2, "ref");
                // printArr(res2.data(), out1, out2, "avx");
            }
            ASSERT_TRUE(equal);
            // printf("%4zu x %4zu, %4zu x %4zu | %7.3f %7.3f | %7.3f | %7.3f vs. peak %7.3f  = %7.4f | %7.5f ms | %7.5f ms | buffer = %zu\n", d1, d2, k1, k2, cpe1, cpe2, cpe1 / cpe2, TOPS/c2, TOPS_PER_CLOCK, (TOPS/c2)/TOPS_PER_CLOCK*100, double(c2)/3e6, t2*1000.0, size_buffer/64);


            printf("\"{%zu x %zu} x {%zu x %zu} x %zu\": { \"in1\": %zu, \"in2\": %zu, \"k1\": %zu, \"k2\": %zu, \"n_kernels\": %zu, \"time\": %7.5f / 1000, \"n_cache_lines_x\": %f, },\n",
                d1, d2, k1, k2, n_kernels, d1, d2, k1, k2, n_kernels, double(c2)/3e6, std::max(1.0f, float(d1 * d2)/64/n_kernels));

            _mm_free(data);
            _mm_free(kernel);
            _mm_free(res1);
            _mm_free(res2);
        }
    }
}


TEST(train_fullyconnected_nn, mnist)
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
