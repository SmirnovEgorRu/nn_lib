
#ifndef __CONV_LAYER_H__
#define __CONV_LAYER_H__

#include "layer_base.h"
#include <immintrin.h>

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

static bool COMPUTE_IN_INT8 = false;

template<typename AType, typename ATypeTmp, typename BType, typename BTypeTmp, typename CType>
void _conv_simple(const AType* frame, const BType* kernel, CType* res, size_t in1, size_t in2, size_t _k1, size_t _k2) {
    size_t out1 = in1 - _k1 + 1;
    size_t out2 = in2 - _k2 + 1;

    for (size_t i = 0; i < out1; ++i) {
        for (size_t j = 0; j < out2; ++j) {
            CType sum = 0;
            for (size_t k1 = 0; k1 < _k1; ++k1) {
                for (size_t k2 = 0; k2 < _k2; ++k2) {
                    sum += ATypeTmp(frame[(i + k1)*in2 + j + k2]) * BTypeTmp(kernel[k1*_k2 + k2]);
                }
            }
            res[i * out2 + j] = sum;
        }
    }
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

// inline __m512i dpbusd(const uint8_t* tmp_x, const int8_t* tmp_k, int32_t* res) {
//     for(size_t i = 0; i < 16; i++) {
//         int16_t sum = 0;
//         for(size_t j = 0; j < 4; j++) {
//             sum += int16_t(tmp_x[i*4 + j]) * int16_t(tmp_k[i*4 + j]);
//         }
//         res[i] = sum;
//     }
// }

inline __m512i dpbusd(__m512i src, const __m512i x, const __m512i k) {
#ifdef __VNNI
    return _mm512_dpbusd_epi32(src, x, k);
#else
    __m256i low_x  = _mm512_extracti32x8_epi32(x, 0);
    __m256i high_x = _mm512_extracti32x8_epi32(x, 1);
    __m256i low_k  = _mm512_extracti32x8_epi32(k, 0);
    __m256i high_k = _mm512_extracti32x8_epi32(k, 1);

    __m512i low16_x  = _mm512_cvtepu8_epi16(low_x);
    __m512i high16_x = _mm512_cvtepu8_epi16(high_x);
    __m512i low16_k  = _mm512_cvtepi8_epi16(low_k);
    __m512i high16_k = _mm512_cvtepi8_epi16(high_k);

    __m512i res1 = _mm512_mullo_epi16(low16_x, low16_k);   // 32 x [16bit integers]
    __m512i res2 = _mm512_mullo_epi16(high16_x, high16_k); // 32 x [16bit integers]

    __m256i idx = _mm256_setr_epi16(0, 1, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m128i r1, r2, r3, r4;
    {
        __m256i tmp = _mm512_extracti32x8_epi32(res1, 0); // 16 x [16bit integers]
        tmp = _mm256_hadd_epi16(tmp, tmp);
        tmp = _mm256_hadd_epi16(tmp, tmp); // 64 * 4
        tmp = _mm256_permutexvar_epi16(idx, tmp); // 4 x 16bit +
        r1 = _mm_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0));
    }
    {
        __m256i tmp = _mm512_extracti32x8_epi32(res1, 1);
        tmp = _mm256_hadd_epi16(tmp, tmp);
        tmp = _mm256_hadd_epi16(tmp, tmp); // 64 * 4
        tmp = _mm256_permutexvar_epi16(idx, tmp); // 4 x 16bit +
        r2 = _mm_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0));
    }
    {
        __m256i tmp = _mm512_extracti32x8_epi32(res2, 0);
        tmp = _mm256_hadd_epi16(tmp, tmp);
        tmp = _mm256_hadd_epi16(tmp, tmp); // 64 * 4
        tmp = _mm256_permutexvar_epi16(idx, tmp); // 4 x 16bit +
        r3 = _mm_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0));
    }
    {
        __m256i tmp = _mm512_extracti32x8_epi32(res2, 1);
        tmp = _mm256_hadd_epi16(tmp, tmp);
        tmp = _mm256_hadd_epi16(tmp, tmp); // 64 * 4
        tmp = _mm256_permutexvar_epi16(idx, tmp); // 4 x 16bit +
        r4 = _mm_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0));
    }

    __m512i res;
    res = _mm512_inserti64x2(res, r1, 0);
    res = _mm512_inserti64x2(res, r2, 1);
    res = _mm512_inserti64x2(res, r3, 2);
    res = _mm512_inserti64x2(res, r4, 3);

    return _mm512_add_epi32(src, res);

    // for(size_t i = 0; i < 16; i++) {
    //     int16_t sum = 0;
    //     for(size_t j = 0; j < 4; j++) {
    //         sum += int16_t(tmp_x[i*4 + j]) * int16_t(tmp_k[i*4 + j]);
    //     }
    //     res[i] = sum;
    // }
#endif
}

// void permutevar(const uint8_t* tmp_x, const uint8_t* idx, const uint8_t* res) {

// }

inline __m512i permutexvar_epi8(__m512i idx, __m512i x) {
#ifdef __VBMI
    x = _mm512_permutexvar_epi8(idx, x);
#else
    __m512i l = _mm512_cvtepu8_epi16(_mm512_extracti32x8_epi32(x, 0));
    __m512i h = _mm512_cvtepu8_epi16(_mm512_extracti32x8_epi32(x, 1));

    __m512i idx_l = _mm512_cvtepu8_epi16(_mm512_extracti32x8_epi32(idx, 0));
    __m512i idx_h = _mm512_cvtepu8_epi16(_mm512_extracti32x8_epi32(idx, 1));

    __mmask32 m1 = _mm512_cmpge_epi16_mask(idx_l, _mm512_set1_epi16(32));
    idx_l = _mm512_mask_sub_epi16(idx_l, m1, idx_l, _mm512_set1_epi16(32));
    idx_l = _mm512_mask_add_epi16(idx_l, m1, idx_l, _mm512_set1_epi16(0x1 << 6));

    __mmask32 m2 = _mm512_cmple_epi16_mask(idx_h, _mm512_set1_epi16(31));
    idx_h = _mm512_mask_sub_epi16(idx_h, _knot_mask32(m2), idx_h, _mm512_set1_epi16(32));

    idx_h = _mm512_mask_add_epi16(idx_h, m2, idx_h, _mm512_set1_epi16(0x1 << 6));

    __m512i lr = _mm512_permutex2var_epi16(l, idx_l, h);
    __m512i hr = _mm512_permutex2var_epi16(l, idx_h, h);

    __m512i res;
    res = _mm512_inserti64x4(res, _mm512_cvtepi16_epi8(lr), 0);
    res = _mm512_inserti64x4(res, _mm512_cvtepi16_epi8(hr), 1);

    return res;
#endif
}

inline __m512i _mm512_loadu_permutexvar_epi8_2idx(const uint8_t* ptr,  const __m512i idx1, const  __m512i idx2) {
    __m512i x16 = _mm512_cvtepu8_epi16(_mm256_loadu_epi8(ptr));

    __m512i lr = _mm512_permutexvar_epi16(x16, idx1);
    __m512i hr = _mm512_permutexvar_epi16(x16, idx2);

    __m512i res;
    res = _mm512_inserti64x4(res, _mm512_cvtepi16_epi8(lr), 0);
    res = _mm512_inserti64x4(res, _mm512_cvtepi16_epi8(hr), 1);

    return res;
}


void _mm512_print_epi32(__m512i x, std::string str = "") {
    printf("%s", str.c_str());
    int32_t arr[16];

    _mm512_storeu_epi32(arr, x);

    for(size_t i = 0; i < 16; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void _mm512_print_epi16(__m512i x, std::string str = "") {
    printf("%s", str.c_str());
    int16_t arr[32];

    _mm512_storeu_epi16(arr, x);

    for(size_t i = 0; i < 32; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void _mm512_print_epi8(__m512i x, std::string str = "") {
    printf("%s", str.c_str());
    int8_t arr[64];

    _mm512_storeu_epi16(arr, x);

    for(size_t i = 0; i < 64; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void vnni_conv(const uint8_t* frame, const int8_t* kernel, int32_t* res, int in1, int in2, int _k1, int _k2) {
    int out1 = in1 - _k1 + 1;
    int out2 = in2 - _k2 + 1;

    // std::fill_n(res, out1 * out2, 0);

    // const __m512i idx = _mm512_setr_epi8(0,   1,  2,  3,  // 1
    //                               1,   2,  3,  4,  // 2
    //                               2,   3,  4,  5,  // 3
    //                               3,   4,  5,  6,  // 4
    //                               4,   5,  6,  7,  // 5
    //                               5,   6,  7,  8,  // 6
    //                               6,   7,  8,  9,  // 7
    //                               7,   8,  9, 10,  // 8
    //                               8,   9, 10, 11,  // 9
    //                               9,  10, 11, 12,  // 10
    //                               10, 11, 12, 13,  // 11
    //                               11, 12, 13, 14,  // 12
    //                               12, 13, 14, 15,  // 13
    //                               13, 14, 15, 16, // 14
    //                               14, 15, 16, 17, // 15
    //                               15, 16, 17, 18); //16

    const __m512i idx = _mm512_set_epi8(18, 17, 16, 15,
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



const __m512i idx1 = _mm512_set_epi16( 10,  9,  8,  7, // 7
                                         9,  8,  7,  6, // 6
                                         8,  7,  6,  5, // 5
                                         7, 6, 5,4, // 4
                                         6,5,4,3,  // 3
                                         5,4,3,2,  // 2
                                         4,3,2,1,  // 1
                                         3,2,1,0); // 0


const __m512i idx2 =   _mm512_set_epi16(18, 17, 16, 15, // 15
     17, 16, 15, 14, // 14
     16, 15, 14, 13, // 13
     15, 14, 13, 12, // 12
     14, 13, 12, 11, // 11
     13, 12, 11, 10, // 10
     12, 11, 10,  9, // 9
     11, 10,  9,  8); // 8



    int n16 = out2 / 16;
    int tail_out = out2 & 0xF;
    // const __mmask16 mask_tail_out = 0xFFFF >> (16-tail_out);

    const __mmask16 mask_tail_out = 0xFFFF >> (16-tail_out);

    int k4 = _k2 / 4;
    int tail_k = _k2 & 3;
    // const __mmask16 mask_tail_out = 0xFFFF >> (16-tail_out);

    int32_t offset_for_tail(0);
    if(tail_k == 1) {
        offset_for_tail = 0xFF;
    } else if (tail_k == 2) {
       offset_for_tail = 0xFFFF;
    } else {
        offset_for_tail = 0xFFFFFF;
    }

    // for(size_t i = 0; i < tail_k; ++i) {
    //     offset_for_tail |= 255 << i; // 255 - 8bit with 1
    // }

    // __m512i* buff = (__m512i*)malloc(sizeof(__m512i) * out2 * in1);

    // __m512i* buff = (__m512i*)alloca(sizeof(__m512i) * out2 * in1);

    // for(size_t i = 0; i < in1; ++i) {
    //     for(size_t j = 0; j < out2; ++j) {
    //         buff[i*out2 + j] = _mm512_loadu_permutexvar_epi8_2idx(frame + i * in2 + j, idx1, idx2);
    //     }
    // }

    for(int irow = 0; irow < out1; ++irow) {
        for(int in16 = 0; in16 < n16; ++in16) {
            int32_t* p = res + irow * out2 + in16*16;
            __m512i r = _mm512_set1_epi32(0);// = _mm512_loadu_epi32(p);

            for(int kk1 = 0; kk1 < _k1; kk1++) {
                int i = 0;
                for(int j = 0; j < k4; ++j, i+=4) {
                    // __m512i x = _mm256_loadu_epi8(frame + (irow + kk1) * in2 + i + in16*16);
                    //x = permutexvar_epi8(idx, x);
                    __m512i x = _mm512_loadu_permutexvar_epi8_2idx(frame + (irow + kk1) * in2 + i + in16*16, idx1, idx2);
                    // __m512i x = buff[(irow + kk1) * out2 + i + in16*16];

                    __m512i k = _mm512_set1_epi32(*((int32_t*)(kernel + kk1*_k2 + i))); // broadcast [4 x 8bit] to 512bit register
                    r = dpbusd(r, x, k);
                }
                if (tail_k) {
                    // __m512i x = _mm512_loadu_epi8(frame + (irow + kk1) * in2 + i + in16*16);
                    // x = permutexvar_epi8(idx, x);
                    __m512i x = _mm512_loadu_permutexvar_epi8_2idx(frame + (irow + kk1) * in2 + i + in16*16, idx1, idx2);

                    // __m512i x = buff[(irow + kk1) * out2 + i + in16*16];

                    int32_t k32 = *((int32_t*)(kernel + kk1*_k2 + i)) & offset_for_tail;
                     __m512i k = _mm512_set1_epi32(k32);
                    r = dpbusd(r, x, k);
                }
            }
            _mm512_storeu_epi32(p, r);
        }
        if (tail_out) {
            int32_t* p = res + irow * out2 + (n16)*16;
            __m512i r = _mm512_set1_epi32(0);// = _mm512_loadu_epi32(p);
            for(int kk1 = 0; kk1 < _k1; kk1++) {
                int i = 0;
                for(int j = 0; j < k4; ++j, i+=4) {
                    // __m512i x = _mm512_loadu_epi8(frame + (irow + kk1) * in2 + i + n16*16);
                    // x = permutexvar_epi8(idx, x);

                    __m512i x = _mm512_loadu_permutexvar_epi8_2idx(frame + (irow + kk1) * in2 + i + n16*16, idx1, idx2);
                    // __m512i x = buff[(irow + kk1) * out2 + i + n16*16];
                    __m512i k = _mm512_set1_epi32(*((int32_t*)(kernel + kk1*_k2 + i))); // broadcast [4 x 8bit] to 512bit register
                    r = dpbusd(r, x, k);
                }
                if (tail_k) {
                    // __m512i x = _mm512_loadu_epi8(frame + (irow + kk1) * in2 + i + n16*16);
                    // x = permutexvar_epi8(idx, x);

                    __m512i x = _mm512_loadu_permutexvar_epi8_2idx(frame + (irow + kk1) * in2 + i + n16*16, idx1, idx2);

                    // __m512i x = buff[(irow + kk1) * out2 + i + n16*16];

                    int32_t k32 = *((int32_t*)(kernel + kk1*_k2 + i)) & offset_for_tail;
                     __m512i k = _mm512_set1_epi32(k32);
                    r = dpbusd(r, x, k);
                }
            }
            _mm512_mask_storeu_epi32(p, mask_tail_out, r);
        }
    }
    // free(buff);
}


void _conv_blocking(const float* frame, const float* kernel, float* res, size_t in1, size_t in2, size_t _k1, size_t _k2) {
    size_t out1 = in1 - _k1 + 1;
    size_t out2 = in2 - _k2 + 1;

    std::fill_n(res, out2 * out2, 0.0f);
    for (size_t k1 = 0; k1 < _k1; ++k1) {
        for (size_t k2 = 0; k2 < _k2; ++k2) {

            size_t irow_begin = k1;
            size_t irow_end = in1 - _k1 + k1 + 1;

            size_t icol_begin = k2;
            size_t icol_end = in2 - _k2 + k2 + 1;

            const float curK = kernel[k1*_k2 + k2];

            for (size_t i = irow_begin; i < irow_end; ++i) {
                for (size_t j = icol_begin; j < icol_end; ++j) {
                    res[(i-irow_begin) * out2 + (j-icol_begin)] += frame[i * in2 + j] * curK;
                }
            }
        }
    }
}


void _conv_blocking_int8(const int8_t* frame, const int8_t* kernel, int32_t* res, size_t in1, size_t in2, size_t _k1, size_t _k2) {
    size_t out1 = in1 - _k1 + 1;
    size_t out2 = in2 - _k2 + 1;

    std::fill_n(res, out2 * out2, 0.0);
    for (size_t k1 = 0; k1 < _k1; ++k1) {
        for (size_t k2 = 0; k2 < _k2; ++k2) {

            size_t irow_begin = k1;
            size_t irow_end = in1 - _k1 + k1 + 1;

            size_t icol_begin = k2;
            size_t icol_end = in2 - _k2 + k2 + 1;

            const float curK = kernel[k1*_k2 + k2];

            for (size_t i = irow_begin; i < irow_end; ++i) {
                for (size_t j = icol_begin; j < icol_end; ++j) {
                    res[(i-irow_begin) * out2 + (j-icol_begin)] += frame[i * in2 + j] * curK;
                }
            }
        }
    }
}

// void _conv_blocking_int8_repack(const int8_t* frame, const int8_t* kernel, int32_t* res, size_t in1, size_t in2, size_t _k1, size_t _k2) {
//     size_t out1 = in1 - _k1 + 1;
//     size_t out2 = in2 - _k2 + 1;

//     int8_t* buffer = (int8_t)alloca(sizeof(int8_t)*_k1*_k2*in1*in2);

//     size_t count = 0;
//     for (size_t i = irow_begin; i < irow_end; ++i) {
//         for (size_t j = icol_begin; j < icol_end; ++j) {
//             auto val = frame[i*in2 + j];
//             for (size_t k1 = 0; k1 < _k1; ++k1) {
//                 for (size_t k2 = 0; k2 < _k2; ++k2) {
//                     buffer[count++] = frame[ (k1 + i)*in2 + j + k2];
//                 }
//             }
//         }
//     }
//     std::fill_n(res, out2 * out2, 0.0);

//     for (size_t k1 = 0; k1 < _k1; ++k1) {
//         for (size_t k2 = 0; k2 < _k2; ++k2) {

//             size_t irow_begin = k1;
//             size_t irow_end = in1 - _k1 + k1 + 1;

//             size_t icol_begin = k2;
//             size_t icol_end = in2 - _k2 + k2 + 1;

//             const float curK = kernel[k1*_k2 + k2];

//             for (size_t i = irow_begin; i < irow_end; ++i) {
//                 for (size_t j = icol_begin; j < icol_end; ++j) {
//                     res[(i-irow_begin) * out2 + (j-icol_begin)] += frame[i * in2 + j] * curK;
//                 }
//             }
//         }
//     }
// }

void _conv_avx512_if(const float* frame, const float* kernel, float* res, size_t in1, size_t in2, size_t _k1, size_t _k2) {
    size_t out1 = in1 - _k1 + 1;
    size_t out2 = in2 - _k2 + 1;

    std::fill_n(res, out2 * out2, 0.0f);
    for (size_t k1 = 0; k1 < _k1; ++k1) {
        for (size_t k2 = 0; k2 < _k2; ++k2) {

            int irow_begin = k1;
            int irow_end = in1 - _k1 + k1 + 1;

            int icol_begin = k2;
            int icol_end = in2 - _k2 + k2 + 1;

            int size = icol_end - icol_begin;

            int n16 = size % 16;
            int tail = size & 15;
            const __mmask16 mask_tail = 0xFFFF >> (16-tail);

            __m512 k = _mm512_set1_ps(kernel[k1*_k2 + k2]);

            if (tail && n16) {
                for (int i = irow_begin; i < irow_end; ++i) {
                    int j = icol_begin;
                    for (; j < icol_end - 15; j+=16) {
                        float* out = res + (i-irow_begin) * out2 + (j-icol_begin);

                        __m512 x = _mm512_loadu_ps(frame + i * in2 + j);
                        __m512 r = _mm512_loadu_ps(out);
                        r = _mm512_fmadd_ps(x, k, r);
                        _mm512_storeu_ps(out, r);
                    }
                    {
                        float* out = res + (i-irow_begin) * out2 + (j-icol_begin);
                        __m512 x = _mm512_maskz_loadu_ps(mask_tail, frame + i * in2 + j);
                        __m512 r = _mm512_maskz_loadu_ps(mask_tail, out);
                        r = _mm512_maskz_fmadd_ps(mask_tail, x, k, r);
                        _mm512_mask_storeu_ps(out, mask_tail, r);
                    }
                }
            } else if (!tail) {
                for (int i = irow_begin; i < irow_end; ++i) {
                    int j = icol_begin;
                    for (; j < icol_end - 15; j+=16) {
                        float* out = res + (i-irow_begin) * out2 + (j-icol_begin);

                        __m512 x = _mm512_loadu_ps(frame + i * in2 + j);
                        __m512 r = _mm512_loadu_ps(out);
                        r = _mm512_fmadd_ps(x, k, r);
                        _mm512_storeu_ps(out, r);
                    }
                }
            } else {
                for (int i = irow_begin; i < irow_end; ++i) {
                    int j = icol_begin;
                    {
                        float* out = res + (i-irow_begin) * out2 + (j-icol_begin);
                        __m512 x = _mm512_maskz_loadu_ps(mask_tail, frame + i * in2 + j);
                        __m512 r = _mm512_maskz_loadu_ps(mask_tail, out);
                        r = _mm512_maskz_fmadd_ps(mask_tail, x, k, r);
                        _mm512_mask_storeu_ps(out, mask_tail, r);
                    }
                }
            }
        }
    }
}

void _conv_avx512(const float* frame, const float* kernel, float* res, int in1, int in2, int _k1, int _k2) {
    int out1 = in1 - _k1 + 1;
    int out2 = in2 - _k2 + 1;

    for (int k1 = 0; k1 < _k1; ++k1) {
        for (int k2 = 0; k2 < _k2; ++k2) {

            int irow_begin = k1;
            int irow_end = in1 - _k1 + k1 + 1;

            int icol_begin = k2;
            int icol_end = in2 - _k2 + k2 + 1;

            int size = icol_end - icol_begin;

            int n16 = size % 16;
            int tail = size & 15;
            const __mmask16 mask_tail = 0xFFFF >> (16-tail);

            __m512 k = _mm512_set1_ps(kernel[k1*_k2 + k2]);

            if (k2 != 0 || k1 !=0) {
                for (int i = irow_begin; i < irow_end; ++i) {
                    int j = icol_begin;
                    for (; j < icol_end - 15; j+=16) {
                        float* out = res + (i-irow_begin) * out2 + (j-icol_begin);
                        __m512 x = _mm512_loadu_ps(frame + i * in2 + j);
                        __m512 r = _mm512_loadu_ps(out);
                        r = _mm512_fmadd_ps(x, k, r);
                        _mm512_storeu_ps(out, r);
                    }
                    {
                        float* out = res + (i-irow_begin) * out2 + (j-icol_begin);
                        __m512 x = _mm512_maskz_loadu_ps(mask_tail, frame + i * in2 + j);
                        __m512 r = _mm512_maskz_loadu_ps(mask_tail, out);
                        r = _mm512_maskz_fmadd_ps(mask_tail, x, k, r);
                        _mm512_mask_storeu_ps(out, mask_tail, r);
                    }
                }
            } else {
                for (int i = irow_begin; i < irow_end; ++i) {
                    int j = icol_begin;
                    for (; j < icol_end - 15; j+=16) {
                        float* out = res + (i-irow_begin) * out2 + (j-icol_begin);
                        __m512 x = _mm512_loadu_ps(frame + i * in2 + j);
                        __m512 r = _mm512_mul_ps(x, k);
                        _mm512_storeu_ps(out, r);
                    }
                    {
                        float* out = res + (i-irow_begin) * out2 + (j-icol_begin);
                        __m512 x = _mm512_maskz_loadu_ps(mask_tail, frame + i * in2 + j);
                        __m512 r = _mm512_maskz_mul_ps(mask_tail, x, k);
                        _mm512_mask_storeu_ps(out, mask_tail, r);
                    }
                }
            }
        }
    }
}


void _conv_avx512_null(const float* frame, const float* kernel, float* res, int in1, int in2, int _k1, int _k2) {
    int out1 = in1 - _k1 + 1;
    int out2 = in2 - _k2 + 1;

    for (int k1 = 0; k1 < _k1; ++k1) {
        for (int k2 = 0; k2 < _k2; ++k2) {
            int irow_begin = k1;
            int irow_end = in1 - _k1 + k1 + 1;

            int icol_begin = k2;
            int icol_end = in2 - _k2 + k2 + 1;

            int size = icol_end - icol_begin;

            int n16 = size % 16;
            int tail = size & 15;
            const __mmask16 mask_tail = 0xFFFF >> (16-tail);

            __m512 k = _mm512_set1_ps(kernel[k1*_k2 + k2]);

            for (int i = irow_begin; i < irow_end; ++i) {
                int j = icol_begin;
                for (; j < icol_end - 15; j+=16) {
                    float* out = res + (i-irow_begin) * out2 + (j-icol_begin);
                    __m512 x = _mm512_loadu_ps(frame + i * in2 + j);
                    __m512 r = _mm512_loadu_ps(out);
                    r = _mm512_fmadd_ps(x, k, r);
                    _mm512_storeu_ps(out, r);
                }
                {
                    float* out = res + (i-irow_begin) * out2 + (j-icol_begin);
                    __m512 x = _mm512_maskz_loadu_ps(mask_tail, frame + i * in2 + j);
                    __m512 r = _mm512_maskz_loadu_ps(mask_tail, out);
                    r = _mm512_maskz_fmadd_ps(mask_tail, x, k, r);
                    _mm512_mask_storeu_ps(out, mask_tail, r);
                }
            }
        }
    }
}

template<typename InType, typename OutType>
class Convolution2d: public LayerBase<InType, OutType> {
    using LayerBase<InType, OutType>::_weights;
    using LayerBase<InType, OutType>::_biases;
    using LayerBase<InType, OutType>::_weightsDerivatives;
    using LayerBase<InType, OutType>::_biasesDerivatives;
    using LayerBase<InType, OutType>::_output;
    using LayerBase<InType, OutType>::_gradient;
public:

    struct Parameter {
        size_t dim1;
        size_t dim2;
        size_t kernel_dim1;
        size_t kernel_dim2;
    };

    Convolution2d(Parameter par, Initializer<InType> initWeigts = Initializer<InType>(), Initializer<InType> initBiases = Initializer<InType>()):
        _par(par), _initWeigts(initWeigts), _initBiases(initBiases) {

        _in1  = _par.dim1;
        _in2  = _par.dim2;
        _k1   = _par.kernel_dim1;
        _k2   = _par.kernel_dim2;
        _out1 = _in1 - _k1 + 1;
        _out2 = _in2 - _k2 + 1;

        _weights.resize(size_t(_k1), size_t(_k2));
        _biases.resize(size_t(1), size_t(1));

        _weightsDerivatives.resize(size_t(_k1), size_t(_k2));
        _biasesDerivatives.resize(size_t(1), size_t(1));

        _initWeigts.generate(_weights.data(), _k1*_k2);
        _initBiases.generate(_biases.data(), 1);
    }

    virtual void forward(const Tensor<InType>& data) {
        auto [in1, in2] = data.getFrameSize();
        ASSERT_TRUE(_in1 == in1);
        ASSERT_TRUE(_in2 == in2);

        _data = &data;
        size_t BatchSize = data.getNBatches();
        _output.resize(BatchSize, _out1, _out2);

        InType* w = _weights.data();

        if (COMPUTE_IN_INT8) {
            std::vector<uint8_t> buff_x(_in1 * _in2 * BatchSize);
            std::vector<int8_t> buff_w(_k1 * _k2);

            const InType* x = data.data();

            InType max =  *std::max_element(x, x + _in1 * _in2*BatchSize);
            InType min =  *std::min_element(x, x + _in1 * _in2*BatchSize);

            InType distance = max - min;
            InType step1 = distance / 255.0f;

            for(size_t i = 0; i < _in1*_in2*BatchSize; ++i) {
                uint8_t tmp = (x[i] / step1);
                buff_x[i] = (x[i] - tmp >= 0.5f ? tmp + 1 : tmp);
            }

            max =  *std::max_element(w, w + _k1 * _k2);
            min =  *std::min_element(w, w + _k1 * _k2);
            InType step2 = std::max(std::abs(max/127.0f), std::abs(min/127.0f));
            for(size_t i = 0; i < _k1*_k2; ++i) {
                int8_t tmp = (w[i] / step2);
                buff_w[i] = (w[i] - tmp >= 0.5f ? tmp + 1 : tmp);
            }

            // max =  *std::max_element(w, w + _k1 * _k2);
            // min =  *std::min_element(w, w + _k1 * _k2);
            // for(size_t i = 0; i < _k1*_k2; ++i) {
            //     InType coeff = (w[i] - min) / (max - min);
            //     std::cout << "tmp: " << coeff << std::endl;
            //     int8_t el = int8_t(uint8_t(coeff * 255.f) - 127u);
            //     std::cout << "el: " << el << std::endl;
            //     buff_w[i] = el;
            // }

            #pragma omp parallel for
            for (size_t iBatch = 0; iBatch < BatchSize; ++iBatch) {
                InType* out =  _output.data() + iBatch* (_out1 * _out2);
                const uint8_t* x   =  buff_x.data() + iBatch* (_in1 * _in2);
                _conv_simple<uint8_t, int16_t, int8_t, int16_t, InType>(x, buff_w.data(), out, _in1, _in2, _k1, _k2);

                for(size_t i = 0; i < _out1*_out2; ++i) {
                    out[i] *= step2 * step1;
                    // out[i] += 127;
                    // out[i] /= 255;
                    // out[i] *= (max - min);
                    // out[i] += min;
                }
            }
        } else {
            #pragma omp parallel for
            for (size_t iBatch = 0; iBatch < BatchSize; ++iBatch) {
                InType* out =  _output.data() + iBatch* (_out1 * _out2);
                const InType* x   =  data.data() + iBatch* (_in1 * _in2);
                // _conv_simple(x, w, out, _in1, _in2, _k1, _k2);

                _conv_avx512(x, w, out, _in1, _in2, _k1, _k2);
            }
        }
    }

    virtual void backward(const Tensor<InType>& grad, bool compute_grad = true) {
        auto [grd1, grd2] = grad.getFrameSize();
        size_t batchSize = grad.getNBatches();

        ASSERT_TRUE(grd1 * grd2 == _out1 * _out2);
        // ASSERT_TRUE(grd2 == _out2);
        ASSERT_TRUE(batchSize == _data->getNBatches());

        if (compute_grad) {
            _gradient.resize(batchSize, _in1, _in2);
            InType* w = _weights.data();

            size_t tmp1 = _out1 + 2 * _k1 - 2;
            size_t tmp2 = _out2 + 2 * _k2 - 2;

            Tensor<InType> tmp(batchSize, tmp1, tmp2);

            #pragma omp parallel for
            for (size_t iBatch = 0; iBatch < batchSize; ++iBatch) {
                const InType* grNext  =  grad.data() + iBatch* (_out1 * _out2);
                InType* g       =  _gradient.data() + iBatch* (_in1 * _in2);
                InType* buff    = tmp.data() + iBatch * tmp1 * tmp2;

                std::fill_n(buff, tmp1 * tmp2, InType(0));
                for (size_t i = 0; i < _out1; ++i) {
                    for (size_t j = 0; j < _out2; ++j) {
                        buff[(i + _k1 -1)*tmp2 + j + _k2  - 1] = grNext[i*_out2 + j];
                    }
                }

                _conv_avx512(buff, w, g, tmp1, tmp2, _k1, _k2);
            }
        }

        computeWeightsDerivatives(batchSize, grad);
    }

    void computeWeightsDerivatives(size_t batchSize, const Tensor<InType>& grad) {
        InType* dw = _weightsDerivatives.data();
        std::fill_n(dw, _k1 * _k2, InType(0));

        for (size_t iBatch = 0; iBatch < batchSize; ++iBatch) {
            const InType* g =  grad.data() + iBatch * (_out1 * _out2);
            const InType* x  =  _data->data() + iBatch* (_in1 * _in2);

            for (size_t i = 0; i < _out1; ++i) {
                for (size_t j = 0; j < _out2; ++j) {
                    InType sum = 0;
                    for (size_t k1 = 0; k1 < _k1; ++k1) {
                        for (size_t k2 = 0; k2 < _k2; ++k2) {
                            dw[k1*_k2 + k2] +=  g[i * _out2 + j] * x[(i + k1)*_in2 + j + k2];
                        }
                    }
                }
            }
        }
        InType alpha = (1.0f / InType(batchSize));
        for (size_t k1 = 0; k1 < _k1; ++k1) {
            for (size_t k2 = 0; k2 < _k2; ++k2) {
                dw[k1*_k2 + k2] *=  alpha;
            }
        }
    }

protected:
    const Tensor<InType>* _data;
    Parameter _par;
    size_t _in1;
    size_t _in2;
    size_t _out1;
    size_t _out2;
    size_t _k1;
    size_t _k2;

    Initializer<InType> _initWeigts;
    Initializer<InType> _initBiases;
};

#endif // __FULLYCONNECTED_LAYER_H__
