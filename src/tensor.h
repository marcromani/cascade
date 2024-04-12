#ifndef CASCADE_TENSOR_H
#define CASCADE_TENSOR_H

#include <cstddef>
#include <vector>

#if CUDA_ENABLED
#define DEFAULT_CPU_VALUE false
#else
#define DEFAULT_CPU_VALUE true
#endif

class Tensor final
{
public:
    Tensor(bool cpu = DEFAULT_CPU_VALUE);
    explicit Tensor(const std::vector<size_t> &shape, bool cpu = DEFAULT_CPU_VALUE);
    explicit Tensor(const std::vector<size_t> &shape, const std::vector<float> &data, bool cpu = DEFAULT_CPU_VALUE);

    ~Tensor();

    size_t size() const;
    const std::vector<size_t> &shape() const;

    template<typename... Args> const float &operator[](Args... indices) const;

    Tensor toCPU() const;
    Tensor toGPU() const;

    Tensor operator+(const Tensor &other) const;

private:
    size_t index(const std::vector<size_t> &indices) const;

    void allocateMemory(size_t size);
    void freeMemory();

    void setData(const std::vector<float> &data);

    void sumCPU(float *result, const float *a, const float *b, size_t size) const;
    void sumGPU(float *result, const float *a, const float *b, size_t size) const;

private:
    bool cpu_;

    std::vector<size_t> shape_;

    float *data_;
    float *hostData_;
};

template<typename... Args> const float &Tensor::operator[](Args... indices) const
{
    // TODO: Static assert to check indices are of type size_t
    const size_t idx = index({static_cast<size_t>(indices)...});
    return data_[idx];
}

#endif
