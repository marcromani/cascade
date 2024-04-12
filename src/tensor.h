#ifndef CASCADE_TENSOR_H
#define CASCADE_TENSOR_H

#include <cstddef>
#include <memory>
#include <vector>

#if CUDA_ENABLED
    #include <cuda_runtime.h>

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

    void setData(const std::vector<float> &data);

    void sumCPU(float *result, const float *a, const float *b, size_t size) const;
    void sumGPU(float *result, const float *a, const float *b, size_t size) const;

private:
    bool cpu_;

    std::vector<size_t> shape_;

    mutable std::shared_ptr<float[]> data_;
    std::shared_ptr<float[]> deviceData_;

    std::vector<Tensor> children_;
    std::vector<Tensor> parents_;
};

template<typename... Args> const float &Tensor::operator[](Args... indices) const
{
    // TODO: Static assert to check indices are of type size_t
    const size_t idx = index({static_cast<size_t>(indices)...});

#if CUDA_ENABLED
    if (data_ == nullptr)
    {
        data_ = std::shared_ptr<float[]>(new float[size()]);
        cudaMemcpy(data_.get(), deviceData_.get(), size() * sizeof(float), cudaMemcpyDeviceToHost);
    }
#endif

    return data_[idx];
}

#endif
