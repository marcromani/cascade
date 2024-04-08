#include "tensor.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>

Tensor::Tensor() : data_(nullptr) {}

Tensor::Tensor(const std::vector<size_t> &shape) : shape_(shape), data_(nullptr)
{
    if (size())
    {
        allocateMemory(size());
    }
}

Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data) : shape_(shape), data_(nullptr)
{
    if (size() != data.size())
    {
        throw std::invalid_argument("Data size does not match tensor shape");
    }

    if (size())
    {
        allocateMemory(size());
        setData(data);
    }
}

Tensor::~Tensor() { freeMemory(); }

size_t Tensor::size() const
{
    if (shape_.empty())
    {
        return 0;
    }

    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
}

const std::vector<size_t> &Tensor::shape() const { return shape_; }

float *Tensor::data() { return data_; }

float &Tensor::operator()(const std::vector<size_t> &indices)
{
    const size_t idx = index(indices);
    return data_[idx];
}

const float &Tensor::operator()(const std::vector<size_t> &indices) const
{
    const size_t idx = index(indices);
    return data_[idx];
}

Tensor Tensor::operator+(const Tensor &other) const
{
    if (shape_ != other.shape_)
    {
        throw std::invalid_argument("Tensor shapes must match for elementwise sum");
    }

    Tensor result(shape_);

#ifdef CUDA_ENABLED
    elementwiseSumGPU(result.data_, data_, other.data_, size());
#else
    elementwiseSumCPU(result.data_, data_, other.data_, size());
#endif

    return result;
}

size_t Tensor::index(const std::vector<size_t> &indices) const
{
    if (indices.size() != shape_.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensionality");
    }

    size_t idx    = 0;
    size_t stride = 1;

    for (int i = indices.size() - 1; i >= 0; --i)
    {
        if (indices[i] >= shape_[i])
        {
            throw std::out_of_range("Index out of range");
        }

        // Row-major order
        idx += indices[i] * stride;
        stride *= shape_[i];
    }

    return idx;
}

void Tensor::allocateMemory(size_t size)
{
#ifdef CUDA_ENABLED
    cudaMallocManaged(static_cast<void **>(reinterpret_cast<void *>(&data_)), size * sizeof(float));
#else
    data_ = new float[size];
#endif
}

void Tensor::freeMemory()
{
#ifdef CUDA_ENABLED
    cudaFree(data_);
#else
    delete[] data_;
#endif
}

void Tensor::setData(const std::vector<float> &data)
{
#ifdef CUDA_ENABLED
    cudaMemcpy(data_, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
#else
    std::copy(data.begin(), data.end(), data_);
#endif
}

void Tensor::elementwiseSumCPU(float *result, const float *a, const float *b, size_t size) const
{
    for (size_t i = 0; i < size; ++i)
    {
        result[i] = a[i] + b[i];
    }
}