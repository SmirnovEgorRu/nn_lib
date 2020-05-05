#ifndef __INITIALIZERS_H__
#define __INITIALIZERS_H__

#include <random>
#include <algorithm>

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
void rand_f(T* ptr, const std::vector<size_t> dims, T start, T end, int seed)
{
    size_t n = 1;
    for(auto elem : dims) {
        n *= elem;
    }

    rand_f(ptr, n, start, end, seed);
}

template<typename T>
class Generator {
public:
    void generate(T* data, size_t n) {
        generateImpl(data, n);
    }

protected:
    virtual void generateImpl(T* data, size_t n) = 0;
};

template<typename T>
class UniformGenerator: public Generator<T> {
public:
    UniformGenerator(size_t seed, T start = -1.0, T end = 1.0):
            _seed(seed), _start(start), _end(end) {
    }

protected:
    size_t _seed;
    T _start;
    T _end;
    virtual void generateImpl(T* data, size_t n) {
        rand_f(data, n, _start, _end, _seed);
    }
};

template<typename T>
class XavierGenerator: public Generator<T> {
public:
    XavierGenerator(size_t seed, size_t n1, size_t n2): _seed(seed), _n1(n1), _n2(n2) { }

protected:
    size_t _seed;
    size_t _n1;
    size_t _n2;
    virtual void generateImpl(T* data, size_t n) {
        T a = std::sqrt(6) / std::sqrt(_n1 + _n2);
        rand_f(data, n, -a, a, _seed);
    }
};

static size_t SEED_DEFAULT = 777;

template<typename T>
class Initializer {
public:

    Initializer(Generator<T>& gen = *(new UniformGenerator<T>(SEED_DEFAULT++))) : _gen(gen){
    }

    ~Initializer() {
        delete &_gen;
    }

    void generate(T* data, size_t n) {
        _gen.generate(data, n);
    }

protected:
    Generator<T>& _gen;
};


// template<typename T>
// class UniformInitializer: public Initializer<T> {
// public:
//     UniformInitializer(size_t seed, T start = -1.0, T end = 1.0):
//             _seed(seed), _start(start), _end(end) {
//     }

// protected:
//     size_t _seed;
//     T _start;
//     T _end;
//     virtual void generateImpl(T* data, size_t n) {
//         rand_f(data, n, _start, _end, _seed)
//     }
// };

#endif // __INITIALIZERS_H__