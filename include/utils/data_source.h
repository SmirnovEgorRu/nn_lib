#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/io.h>
// #include <tbb/tbb.h>
// #include <daal.h>
#include <stdlib.h>
#include <omp.h>
#include <tbb/scalable_allocator.h>
#include <cstring>

namespace fast_data_source
{

extern "C" {
    float __FPK_string_to_float(const char * nptr, char ** endptr);
    double __FPK_string_to_double(const char * nptr, char ** endptr);
}

size_t get_size(const char* name)
{
    FILE * ptrFile = fopen(name, "rb" );
    fseek(ptrFile , 0 , SEEK_END);
    size_t size = ftell(ptrFile);
    rewind (ptrFile);
    fclose (ptrFile);
    return size;
}

float* resize(float* ptr, const size_t prev_size)
{
    float* new_ptr = (float*)scalable_malloc(sizeof(float) * prev_size * 2);
    std::memcpy(new_ptr, ptr, sizeof(float) * prev_size);
    scalable_free(ptr);
    return new_ptr;
}

inline bool isNumber(const char ch)
{
    return ch != ',' && ch != '\n';
}

inline bool isDelimiter(const char ch)
{
    return ch == ',' || ch == '\n';
}

inline bool isLineEnd(const char ch)
{
    return ch == '\n';
}

size_t read_csv(float** data, float** labels, size_t n_feat, const char* name)
{
    const size_t size = get_size(name);
    int fdin = open(name, O_RDONLY);
    char* src = (char*)mmap(0, size, PROT_READ, MAP_SHARED, fdin, 0);

    const size_t block_size = 2048;
    size_t n_blocks = size/block_size;
    n_blocks += !!(size - n_blocks*block_size);

    std::vector<std::pair<float*,size_t>> v(n_blocks);

    #pragma omp parallel for
    for(size_t j = 0; j < n_blocks; ++j)
    {
        size_t iStart = j * block_size;
        size_t iEnd = ( (j + 1)  * block_size) > size ? size : ( (j + 1)  * block_size);

        if (j)
        {
            while(!isLineEnd(src[iStart-1]) && iStart < size) iStart++;
        }
        while(!isLineEnd(src[iEnd-1]) && iEnd < size) iEnd++;

        size_t mem_size = 1024;
        float* ptr = (float*)scalable_malloc(sizeof(float) * mem_size);

        size_t prev = iStart;
        size_t n_num = 0;
        size_t i = iStart;

        for(; i < iEnd; i++)
        {
            if (isDelimiter(src[i]))
            {
                char* tmp = src + i;
                ptr[n_num++] = __FPK_string_to_float(src + prev, &tmp);
                prev = i + 1;
                if (n_num > mem_size)
                {
                    ptr = resize(ptr, mem_size);
                    mem_size *= 2;
                }
            }
        }

        v[j].first = ptr;
        v[j].second = n_num;
    }
    size_t nrows = 0;

    std::vector<size_t> rows(n_blocks);
    std::vector<size_t> row_offset(n_blocks);

    for(size_t i = 0; i < n_blocks; ++i)
    {
        rows[i] = v[i].second / (n_feat+1);
        row_offset[i] = nrows;
        nrows += rows[i];

    }
    *data   = (float*)scalable_malloc(sizeof(float) * nrows * n_feat);
    *labels = (float*)scalable_malloc(sizeof(float) * nrows);

    #pragma omp parallel for
    for(size_t j = 0; j < n_blocks; ++j)
    {
        float* local_data = *data + row_offset[j] * n_feat;
        float* local_label = *labels + row_offset[j];

        for(size_t i = 0; i < rows[j]; ++i)
        {
            for(size_t k = 0; k < n_feat; ++k)
                local_data[i*n_feat + k] = v[j].first[i * (n_feat+1) + k];

            local_label[i] = v[j].first[i * (n_feat+1) + n_feat];
        }
        scalable_free(v[j].first);
    }

    return nrows;
}
}