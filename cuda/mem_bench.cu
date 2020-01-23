
#include "stdio.h"
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iostream>
#include "omp.h"
#include "immintrin.h"

size_t N_THREADS = 1;

struct TimeInfo{
    double min;
    double median;
    double avg;
};

template<typename T>
void rand_f(T* ptr, size_t n, T start, T end, int seed)
{
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<T> distribution(start, end);
    std::generate(ptr, ptr + n, [&]()
    {
        return distribution(generator);
    });
}

template<typename T>
void printArr(const T* data, size_t n, std::string str = std::string()) {
    std::cout << str <<  ": [ ";
    for(size_t i = 0; i < n; ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << " ]" << std::endl;
}

template<typename T>
void printArr(const T* data, size_t dim1, size_t dim2, std::string str = std::string()) {
    std::cout << str <<  ":\n[ \n";
    for(size_t i = 0; i <  dim1; ++i) {
        std::cout << "[";
        for(size_t j = 0; j < dim2; ++j) {
            std::cout << data[i * dim2 + j] << ", ";
        }
        std::cout << "], \n";
    }
    std::cout << " ]" << std::endl;
}

void conv_simple(const float* frame, const float* kernel, float* res, size_t in1, size_t in2, size_t _k1, size_t _k2) {
    size_t out1 = in1 - _k1 + 1;
    size_t out2 = in2 - _k2 + 1;

    for (size_t i = 0; i < out1; ++i) {
        for (size_t j = 0; j < out2; ++j) {
            float sum = 0;
            for (size_t k1 = 0; k1 < _k1; ++k1) {
                for (size_t k2 = 0; k2 < _k2; ++k2) {
                    sum += frame[(i + k1)*in2 + j + k2] * kernel[k1*_k2 + k2];
                }
            }
            res[i * out2 + j] = sum;
        }
    }
}

template<typename Func>
TimeInfo  measure(Func func, std::string str, size_t nIter = 50)
{
    std::vector<double> v(nIter);
    std::chrono::time_point<std::chrono::steady_clock> t1, t2;

    for(size_t i = 0; i < nIter; i++)
    {
        t1 = std::chrono::steady_clock::now();
        func();
        t2 = std::chrono::steady_clock::now();
        v[i] = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
    }

    double avg = 0;
    for(auto i = v.begin() ; i != v.end(); i++) {
        avg += *i;
    }
    avg /= v.size();
    std::sort(v.begin(),  v.end());


    printf("%s | Min = %9.5f | Med = %9.5f | Avg = %9.5f\n", str.c_str(), v[0], v[nIter/2], avg);
    return {v[0], v[nIter/2], avg};
}

__global__ void image_convolution_kernel(float *input, float *out, float *kernelConv,
                    int img_width, const int img_height,
                    const int kernel_width, const int kernel_height )
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x < img_width) && (y < img_height)){  // thread check
      float sum = 0;
      for ( int j = 0; j < kernel_height; j++ )
      {
        for ( int i = 0; i < kernel_width; i++ )
        {
            int dX = x + i - kernel_width / 2;
            int dY = y + j - kernel_height / 2;

            if ( dX < 0 )
                dX = 0;

            if ( dX >= img_width )
                dX = img_width - 1;

            if ( dY < 0 )
                dY = 0;

            if ( dY >= img_height )
                dY = img_height - 1;


            const int idMat = j * kernel_width + i;
            const int idPixel = dY * img_width + dX;
            sum += (float)input[idPixel] * kernelConv[idMat];
        }
      }

      const int idOut = y * img_width + x;
      out[idOut] = abs(sum);
    }

}

void convolution(float * input, float* output, float* kernel, size_t in1, size_t in2, size_t k1, size_t k2)
{
    size_t out1 = in1 - k1 + 1;
    size_t out2 = in2 - k2 + 1;

    float * d_input, * d_output, * d_kernel;
    cudaMalloc(&d_input, in1*in2*sizeof(float));
    cudaMalloc(&d_output, out1*out2*sizeof(float));
    cudaMalloc(&d_kernel, k1*k2*sizeof(float));

    cudaMemcpy(d_input, input, in1*in2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, in1*in2*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocksize(16,16);
    dim3 gridsize;
    gridsize.x=(in2+blocksize.x-1)/blocksize.x;
    gridsize.y=(in1+blocksize.y-1)/blocksize.y;

    image_convolution_kernel<<<gridsize,blocksize>>>(d_input,d_output,d_kernel,in2,in1,k2,k1);
    cudaMemcpy(output, d_output, out1*out2*sizeof(float), cudaMemcpyDeviceToHost);
}


void convolution_device_mem(float * d_input, float* d_output, float* d_kernel, size_t in1, size_t in2, size_t k1, size_t k2)
{
    // size_t out1 = in1 - k1 + 1;
    // size_t out2 = in2 - k2 + 1;

    dim3 blocksize(16,16);
    dim3 gridsize;
    gridsize.x=(in2+blocksize.x-1)/blocksize.x;
    gridsize.y=(in1+blocksize.y-1)/blocksize.y;

    image_convolution_kernel<<<gridsize,blocksize>>>(d_input,d_output,d_kernel,in2,in1,k2,k1);
}

__global__ void conv_kernel_2(const float* data, const float* kernel, float* out,
        const size_t in1, const size_t in2, const size_t k1, const size_t k2, const size_t out1, const size_t out2)
{
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if(j < out2 && i < out1) {
        size_t idx = i * out2 + j;
        out[idx] = 0.0f;
        for( int _k1 = 0; _k1 < k1; ++_k1) {
            for( int _k2 = 0; _k2 < k2; ++_k2 ) {
                out[idx] += data[(i + _k1)*in2 + j + _k2] * kernel[_k1*k2 + _k2];
            }
        }
    }
}

__global__ void conv_kernel_batched_2(const float* data, const float* kernel, float* out,
        const size_t in1, const size_t in2, const size_t k1, const size_t k2, const size_t out1, const size_t out2, size_t batches)
{
    const int n_blocks_per_frame = (gridDim.x / batches) * (gridDim.y / batches);

    const int i_block  = blockIdx.y * gridDim.x + blockIdx.x;
    const int fram_num = i_block / n_blocks_per_frame;

    const int idx_of_block_in_frame = i_block % n_blocks_per_frame;

    const int y_blocks = (out2 / blockDim.y) + !!(out2 % blockDim.y);
    const int x_blocks = (out1 / blockDim.x) + !!(out1 % blockDim.x);

    const int blockIdx_y = idx_of_block_in_frame % y_blocks;
    const int blockIdx_x = idx_of_block_in_frame / x_blocks;

    const int i = blockIdx_y * blockDim.y + threadIdx.y;
    const int j = blockIdx_x * blockDim.x + threadIdx.x;

    // printf(" %d %d | %d %d | %d %d | %d %d %d | %d %d\n", int(blockIdx.x), int(blockIdx.y), int(i_block), int(idx_of_block_in_frame), int(blockDim.x), int(blockDim.y), int(fram_num), int(n_blocks_per_frame), int(i_block), int(i), int(j));
    if(j < out2 && i < out1 && fram_num < batches) {
        // printf("%d %d %d\n", int(fram_num), int(i), int(j));
        size_t idx = i * out2 + j + fram_num * out1*out2;
        out[idx] = 0.0f;
        for( int _k1 = 0; _k1 < k1; ++_k1) {
            for( int _k2 = 0; _k2 < k2; ++_k2 ) {
                out[idx] += data[fram_num*in1*in2 + (i + _k1)*in2 + j + _k2] * kernel[_k1*k2 + _k2];
            }
        }
    }
}

__global__ void conv_kernel_batched_vec_2(const float* data, const float* kernel, float* out,
        const size_t in1, const size_t in2, const size_t k1, const size_t k2, const size_t out1, const size_t out2, size_t batches)
{
    const int fram_num = threadIdx.y;

    const int i = threadIdx.x % out2;
    const int j = threadIdx.x / out1;

    size_t idx = i * out2 + j + fram_num * out1*out2;
    out[idx] = 0.0f;
    for( int _k1 = 0; _k1 < k1; ++_k1) {
        for( int _k2 = 0; _k2 < k2; ++_k2 ) {
            out[idx] += data[fram_num*in1*in2 + (i + _k1)*in2 + j + _k2] * kernel[_k1*k2 + _k2];
        }
    }
}

void test_convolution_device() {
    size_t d1 = 28;
    size_t d2 = 28;
    size_t k1 = 5;
    size_t k2 = 5;

    size_t out1 = d1 - k1 + 1;
    size_t out2 = d2 - k2 + 1;
    size_t batchSize = 512;

    std::vector<float> data(d1 * d2 * batchSize);
    std::vector<float> out(out1 * out2 * batchSize);
    std::vector<float> kernel(k1 * k2);
    std::vector<float> ref(out1 * out2 * batchSize);

    rand_f<float>(data.data(), data.size(), -1.0, 1.0, 111);
    rand_f<float>(kernel.data(), kernel.size(), -1, 1, 112);

    rand_f<float>(out.data(), out.size(), 0, 0, 112);
    rand_f<float>(ref.data(), ref.size(), 0, 0, 112);

    float * d_input, * d_output, * d_kernel;
    cudaMalloc(&d_input, batchSize*d1*d2*sizeof(float));
    cudaMalloc(&d_kernel, k1*k2*sizeof(float));
    cudaMalloc(&d_output, batchSize*out1*out2*sizeof(float));

    cudaMemcpy(d_input, data.data(), batchSize*d1*d2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), k1*k2*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocksize(16, 16);
    dim3 gridsize;
    gridsize.x = (out1 / blocksize.x + !!(out1 % blocksize.x)) * batchSize;
    gridsize.y = (out2 / blocksize.y + !!(out2 % blocksize.y)) * batchSize;

    printf("gridsize.x = %u | gridsize.y = %u\n", gridsize.x, gridsize.y);

    measure([&]() {
        #pragma omp parallel for
        for (size_t iBatch = 0; iBatch < batchSize; ++iBatch) {
            float* r =  ref.data() + iBatch* (out1 * out2);
            float* x =  data.data() + iBatch* (d1 * d2);
            conv_simple(x, kernel.data(), r, d1, d2, k1, k2);
        }
    }, "CPU conv         ", 10);

    measure([&]() {
        conv_kernel_batched_2<<<gridsize,blocksize>>>(d_input, d_kernel, d_output, d1, d2, k1, k2, out1, out2, batchSize);
        cudaThreadSynchronize();
    }, "cuda conv sync   ", 5);

    cudaMemcpy(out.data(), d_output, batchSize*out1*out2*sizeof(float), cudaMemcpyDeviceToHost);
    if (!std::equal(out.begin(), out.end(), ref.begin(), [](float r1, float r2) {
        return std::isfinite(r1) && std::isfinite(r2) && std::abs(r1 - r2) < 0.001;
    })) { printf("Failed!\n"); return; }

    measure([&]() {
        conv_kernel_batched_vec_2<<<out1*out2, batchSize>>>(d_input, d_kernel, d_output, d1, d2, k1, k2, out1, out2, batchSize);
        cudaThreadSynchronize();
    }, "cuda conv vec1   ", 10);

    cudaMemcpy(out.data(), d_output, batchSize*out1*out2*sizeof(float), cudaMemcpyDeviceToHost);
    if (!std::equal(out.begin(), out.end(), ref.begin(), [](float r1, float r2) {
        return std::isfinite(r1) && std::isfinite(r2) && std::abs(r1 - r2) < 0.001;
    })) { printf("Failed!\n"); return; }
}



int main(int argc, char* argv[]) {
    // if (argc > 1)
    // {
    //     N_THREADS = std::atoi(argv[1]);
    // }

    // test_convolution();

    test_convolution_device();
}
