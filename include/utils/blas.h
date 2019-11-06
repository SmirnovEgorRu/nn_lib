
#ifndef __BLAS_H__
#define __BLAS_H__

#include "mkl.h"

// template<typename T>
// void gemm(bool TransA, bool TransB, size_t m, size_t n, size_t k, T alpha, const T* a, size_t lda, const T* b, size_t ldb, T beta, T* c, size_t ldc) {
//     T* newA = (T*)a;
//     T* newB = (T*)b;

//     if(TransA) {
//         T* buff = (T*)malloc(m * k * sizeof(T)); // TODO: add gemm for transpose data
//         transpose(a, buff, m, k);
//         newA = buff;
//     }
//     if(TransB) {
//         T* buff = (T*)malloc(n * k * sizeof(T)); // TODO: add gemm for transpose data
//         transpose(b, buff, n, k);
//         newB = buff;
//     }

//     for(size_t i = 0; i < m; ++i) {
//         if (!beta) {
//             std::fill_n(c + i * ldc, k, 0.0f);
//         } else {
//             for(size_t j = 0 ; j < k; ++j) {
//                 c[i * ldc + j] *= beta;
//             }
//         }
//         for(size_t j = 0; j < n; ++j) {
//             const float a_const = alpha * newA[i*lda + j];
//             for(size_t p = 0 ; p < k; ++p) {
//                 c[i * ldc + p] += a_const * newB[j*ldb + p];
//             }
//         }
//     }
//     if(TransA) free(newA);
//     if(TransB) free(newB);
// }

// template<typename T>
static inline void gemm(bool TransA, bool TransB, size_t m, size_t n, size_t k, float alpha, const float* a, size_t lda, const float* b, size_t ldb, float beta, float* c, size_t ldc) {
    cblas_sgemm(CblasRowMajor,
                TransA ? CblasTrans : CblasNoTrans,
                TransB ? CblasTrans : CblasNoTrans,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc);
}

static inline void gemm(bool TransA, bool TransB, size_t m, size_t n, size_t k, double alpha, const double* a, size_t lda, const double* b, size_t ldb, double beta, double* c, size_t ldc) {
    cblas_dgemm(CblasRowMajor,
                TransA ? CblasTrans : CblasNoTrans,
                TransB ? CblasTrans : CblasNoTrans,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc);
}

template<typename T>
static inline void transpose(const T* in, T* out, size_t n, size_t m) {
    for(size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < m; j++) {
            out[j*n + i] = in[i*m +j];
        }
    }
}

template<typename T>
static inline void NnExp(size_t n, T* in, T* out) {
    ASSERT_TRUE(0);
}

static inline void NnExp(size_t n, float* in, float* out) {
    vsExp(n, in, out);
}

static inline void NnExp(size_t n, double* in, double* out) {
    vdExp(n, in, out);
}

#endif // #ifndef __BLAS_H__
