#include "tensor.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#if CUDA_ENABLED
    #include <cuda_runtime.h>
#endif

Tensor::Tensor(bool cpu) : cpu_(cpu), data_(nullptr), deviceData_(nullptr) {}

Tensor::Tensor(const std::vector<size_t> &shape, bool cpu)
: cpu_(cpu)
, shape_(shape)
, data_(nullptr)
, deviceData_(nullptr)
{
    if (size() > 0)
    {
        allocateMemory(size());
    }
}

Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data, bool cpu)
: cpu_(cpu)
, shape_(shape)
, data_(nullptr)
, deviceData_(nullptr)
{
    if (data.size() != size())
    {
        throw std::invalid_argument("Data size does not match tensor shape");
    }

    if (size() > 0)
    {
        allocateMemory(size());
        setData(data);
    }
}

Tensor::~Tensor() {}

size_t Tensor::size() const
{
    if (shape_.empty())
    {
        return 0;
    }

    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
}

const std::vector<size_t> &Tensor::shape() const { return shape_; }

Tensor Tensor::toCPU() const
{
#if CUDA_ENABLED
    if (cpu_)
    {
        return *this;
    }
    else
    {
        float *tmp = new float[size()];
        cudaMemcpy(tmp, deviceData_.get(), size() * sizeof(float), cudaMemcpyDeviceToHost);

        const std::vector<float> data(tmp, tmp + size());
        delete[] tmp;

        return Tensor(shape_, data, true);
    }
#else
    return *this;
#endif
}

Tensor Tensor::toGPU() const
{
    if (!cpu_)
    {
        return *this;
    }
    else
    {
        const std::vector<float> data(data_.get(), data_.get() + size());

        return Tensor(shape_, data, false);
    }
}

Tensor Tensor::operator+(const Tensor &other) const
{
    if (other.shape_ != shape_)
    {
        throw std::invalid_argument("Tensor shapes must match for elementwise sum");
    }

    // TODO: All tensors should be in the GPU or the CPU
    Tensor result(shape_);

#if CUDA_ENABLED
    if (cpu_)
    {
        sumCPU(result.data_.get(), data_.get(), other.data_.get(), size());
    }
    else
    {
        sumGPU(result.deviceData_.get(), deviceData_.get(), other.deviceData_.get(), size());
    }
#else
    sumCPU(result.data_.get(), data_.get(), other.data_.get(), size());
#endif

    return result;
}

size_t Tensor::index(const std::vector<size_t> &indices) const
{
    // No need to check if indices is empty as array subscripting requires at least one parameter
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
#if CUDA_ENABLED
    if (cpu_)
    {
        data_ = std::shared_ptr<float[]>(new float[size]);
    }
    else
    {
        float *tmp;
        cudaMalloc(reinterpret_cast<void **>(&tmp), size * sizeof(float));

        auto cudaDeleter = [](float *ptr) { cudaFree(ptr); };
        deviceData_      = std::shared_ptr<float[]>(tmp, cudaDeleter);
    }
#else
    data_ = std::shared_ptr<float[]>(new float[size]);
#endif
}

void Tensor::setData(const std::vector<float> &data)
{
#if CUDA_ENABLED
    if (cpu_)
    {
        std::copy(data.begin(), data.end(), data_.get());
    }
    else
    {
        cudaMemcpy(deviceData_.get(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
#else
    std::copy(data.begin(), data.end(), data_.get());
#endif
}

void Tensor::sumCPU(float *result, const float *a, const float *b, size_t size) const
{
    for (size_t i = 0; i < size; ++i)
    {
        result[i] = a[i] + b[i];
    }
}
