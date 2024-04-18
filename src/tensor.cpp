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

namespace cascade
{
void sumForward(const Tensor &result, const Tensor &x, const Tensor &y);
void sumBackward(const Tensor &x, const Tensor &y);

Tensor::Tensor(bool device) : device_(device)
{
#if !CUDA_ENABLED
    device_ = false;
#endif

    hostDataNeedsUpdate_   = false;
    deviceDataNeedsUpdate_ = device_;
}

Tensor::Tensor(const std::vector<size_t> &shape, bool device) : device_(device), shape_(shape)
{
#if !CUDA_ENABLED
    device_ = false;
#endif

    hostDataNeedsUpdate_   = false;
    deviceDataNeedsUpdate_ = device_;

    size_t n = size();

    if (n > 0)
    {
        allocateMemory(n, false);

        if (device_)
        {
            // TODO: Write a kernel to initialize the data
        }
        else
        {
            std::fill(hostData_.get(), hostData_.get() + n, 0.f);
        }
    }
}

Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data, bool device)
: device_(device)
, shape_(shape)
{
#if !CUDA_ENABLED
    device_ = false;
#endif

    hostDataNeedsUpdate_   = false;
    deviceDataNeedsUpdate_ = device_;

    size_t n = size();

    if (data.size() != n)
    {
        throw std::invalid_argument("Data size is inconsistent with tensor shape");
    }

    if (n > 0)
    {
        allocateMemory(n, false);
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

        tensor.allocateMemory(n, false);
        cudaMemcpy(tensor.hostData_.get(), deviceData_.get(), n * sizeof(float), cudaMemcpyDeviceToHost);

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

    tensor.allocateMemory(n, false);
    cudaMemcpy(tensor.deviceData_.get(), hostData_.get(), n * sizeof(float), cudaMemcpyHostToDevice);

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

    size_t n = size();

    // TODO: Maybe we should do lazy allocation too
    x.allocateMemory(n * n, true);
    y.allocateMemory(n * n, true);

    Tensor result(shape_, true);

    result.children_.push_back(x);
    result.children_.push_back(y);

#if CUDA_ENABLED
    if (result.device_)
    {
        result.forward_  = [result, x, y]() { kernelSumForward(result, x, y); };
        result.backward_ = [result, x, y]() { kernelSumBackward(x, y); };
    }
    else
    {
        result.forward_  = [result, x, y]() { sumForward(result, x, y); };
        result.backward_ = [result, x, y]() { sumBackward(x, y); };
    }
#else
    result.forward_  = [result, x, y]() { sumForward(result, x, y); };
    result.backward_ = [result, x, y]() { sumBackward(x, y); };
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

void Tensor::allocateMemory(size_t size, bool grad)
{
#if CUDA_ENABLED
    if (device_)
    {
        float *tmp;
        cudaMalloc(reinterpret_cast<void **>(&tmp), size * sizeof(float));

        auto cudaDeleter = [](float *p) { cudaFree(p); };

        if (grad)
        {
            deviceGrad_ = std::shared_ptr<float[]>(tmp, cudaDeleter);
        }
        else
        {
            deviceData_ = std::shared_ptr<float[]>(tmp, cudaDeleter);
        }
    }
    else
    {
        if (grad)
        {
            hostGrad_ = std::shared_ptr<float[]>(new float[size]);
        }
        else
        {
            hostData_ = std::shared_ptr<float[]>(new float[size]);
        }
    }
#else
    if (grad)
    {
        hostGrad_ = std::shared_ptr<float[]>(new float[size]);
    }
    else
    {
        hostData_ = std::shared_ptr<float[]>(new float[size]);
    }
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
        std::copy(data.begin(), data.end(), hostData_.get());
    }
#else
    std::copy(data.begin(), data.end(), hostData_.get());
#endif
}

void sumForward(const Tensor &result, const Tensor &x, const Tensor &y)
{
    for (size_t i = 0; i < result.size(); ++i)
    {
        result.hostData_[i] = x.hostData_[i] + y.hostData_[i];
    }
}

void sumBackward(const Tensor &x, const Tensor &y)
{
    // TODO
}
}  // namespace cascade
