#include "tensor.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#if CUDA_ENABLED
#include "kernels/kernel_sum.h"

#include <cuda_runtime.h>
#endif

void sumForward(float *result, const float *x, const float *y, size_t size);
void sumBackward(float *result, const float *x, const float *y, size_t size);

namespace cascade
{
Tensor::Tensor(bool device) : data_(nullptr), deviceData_(nullptr), grad_(nullptr), deviceGrad_(nullptr)
{
#if CUDA_ENABLED
    device_ = device;
#else
    [&device] {}();  // Silence the unused parameter warning
    device_ = false;
#endif
}

Tensor::Tensor(const std::vector<size_t> &shape, bool device)
: shape_(shape)
, data_(nullptr)
, deviceData_(nullptr)
, grad_(nullptr)
, deviceGrad_(nullptr)
{
#if CUDA_ENABLED
    device_ = device;
#else
    [&device] {}();  // Silence the unused parameter warning
    device_ = false;
#endif

    size_t n = size();

    if (n > 0)
    {
        allocateMemory(data_, n);

        if (device_)
        {
            // TODO: Write a kernel to initialize the data
        }
        else
        {
            std::fill(data_.get(), data_.get() + n, 0.f);
        }
    }
}

Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data, bool device)
: shape_(shape)
, data_(nullptr)
, deviceData_(nullptr)
, grad_(nullptr)
, deviceGrad_(nullptr)
{
#if CUDA_ENABLED
    device_ = device;
#else
    [&device] {}();  // Silence the unused parameter warning
    device_ = false;
#endif

    size_t n = size();

    if (data.size() != n)
    {
        throw std::invalid_argument("Data size is inconsistent with tensor shape");
    }

    if (n > 0)
    {
        allocateMemory(data_, n);
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

Tensor Tensor::toHost() const
{
#if CUDA_ENABLED
    if (device_)
    {
        Tensor tensor(shape_, false);

        size_t n = size();

        tensor.allocateMemory(tensor.data_, n);
        cudaMemcpy(tensor.data_.get(), deviceData_.get(), n * sizeof(float), cudaMemcpyDeviceToHost);

        return tensor;
    }
    else
    {
        return *this;
    }
#else
    return *this;
#endif
}

Tensor Tensor::toDevice() const
{
    if (device_)
    {
        return *this;
    }

#if CUDA_ENABLED
    Tensor tensor(shape_, true);

    size_t n = size();

    tensor.allocateMemory(tensor.data_, n);
    cudaMemcpy(tensor.deviceData_.get(), data_.get(), n * sizeof(float), cudaMemcpyHostToDevice);

    return tensor;
#else
    return *this;
#endif
}

Tensor Tensor::operator+(const Tensor &other) const
{
    if (other.shape_ != shape_)
    {
        throw std::invalid_argument("Tensor shapes must match for elementwise sum");
    }

    Tensor x = toDevice();
    Tensor y = other.toDevice();

    // TODO: Reserve x and y grad or deviceGrad (depending on CUDA_ENABLED)
    // of appropriate size, so that kernelSumBackward can fill them with gradients

    x.allocateGradMemory(size() * size());
    y.allocateGradMemory(size() * size());

    Tensor result(shape_, false);

    result.children_.push_back(x);
    result.children_.push_back(y);

#if CUDA_ENABLED
    if (result.cpu_)
    {
        result.forward_
            = [result, x, y]() { sumForward(result.data_.get(), x.data_.get(), y.data_.get(), result.size()); };
    }
    else
    {
        result.forward_ = [result, x, y]()
        { kernelSumForward(result.deviceData_.get(), x.deviceData_.get(), y.deviceData_.get(), result.size()); };

        result.backward_ = [result, x, y]()
        {
            size_t *shape;
            cudaMalloc(reinterpret_cast<void **>(&shape), (x.shape().size() + 1) * sizeof(size_t));

            const size_t numDims = x.shape().size();
            cudaMemcpy(shape, &numDims, sizeof(size_t), cudaMemcpyHostToDevice);

            cudaMemcpy(shape + 1, x.shape().data(), x.shape().size() * sizeof(size_t), cudaMemcpyHostToDevice);

            kernelSumBackward(x.deviceGrad_.get(), y.deviceGrad_.get(), x.size(), shape);

            cudaFree(shape);
        };
    }
#else
    result.forward_ = [result, x, y]() { sumForward(result.data_.get(), x.data_.get(), y.data_.get(), result.size()); };
#endif

    return result;
}

size_t Tensor::index(const std::vector<size_t> &indices) const
{
    size_t n = indices.size();

    // No need to check if indices is empty as array subscripting requires at least one parameter
    if (n != shape_.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensionality");
    }

    size_t idx    = 0;
    size_t stride = 1;

    for (int i = n - 1; i >= 0; --i)
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

void Tensor::allocateMemory(std::shared_ptr<float[]> &ptr, size_t size)
{
#if CUDA_ENABLED
    if (device_)
    {
        float *tmp;
        cudaMalloc(reinterpret_cast<void **>(&tmp), size * sizeof(float));

        auto cudaDeleter = [](float *p) { cudaFree(p); };
        ptr              = std::shared_ptr<float[]>(tmp, cudaDeleter);
    }
    else
    {
        ptr = std::shared_ptr<float[]>(new float[size]);
    }
#else
    ptr             = std::shared_ptr<float[]>(new float[size]);
#endif
}

void Tensor::setData(const std::vector<float> &data)
{
#if CUDA_ENABLED
    if (device_)
    {
        cudaMemcpy(deviceData_.get(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    else
    {
        std::copy(data.begin(), data.end(), data_.get());
    }
#else
    std::copy(data.begin(), data.end(), data_.get());
#endif
}
}  // namespace cascade

void sumForward(float *result, const float *x, const float *y, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        result[i] = x[i] + y[i];
    }
}

void sumBackward(float *result, const float *x, const float *y, size_t size)
{
    // TODO
}
