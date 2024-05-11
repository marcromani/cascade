#include "tensor.h"

#include "tensor_data.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <ostream>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if CUDA_ENABLED
#include "kernels/kernel_add.h"
#include "kernels/kernel_mul.h"

#include <cuda_runtime.h>
#endif

namespace cascade
{
Tensor::Tensor() : scalar_(false), data_(std::make_shared<TensorData>())
{
    data_->device = false;

#if CUDA_ENABLED
    data_->hostDataNeedsUpdate   = false;
    data_->deviceDataNeedsUpdate = false;
#endif
}

Tensor::Tensor(const std::vector<size_t> &shape, [[maybe_unused]] bool device)
: scalar_(shape.empty())
, sliceShape_(shape)
, sliceOffset_(std::vector<size_t>(std::max(shape.size(), static_cast<size_t>(1)), 0))
, data_(std::make_shared<TensorData>())
{
#if CUDA_ENABLED
    data_->device = device;

    data_->hostDataNeedsUpdate   = data_->device;
    data_->deviceDataNeedsUpdate = false;
#else
    data_->device    = false;
#endif

    data_->shape = shape;

    size_t n = size();

    if (n > 0)
    {
        allocateMemory(n, false);

        if (data_->device)
        {
            // TODO: Write a kernel to initialize the data
        }
        else
        {
            // std::fill(data_->hostData.get(), data_->hostData.get() + n, 0.f);
        }
    }
}

Tensor::Tensor(const std::initializer_list<size_t> &shape, bool device) : Tensor(std::vector<size_t>(shape), device) {}

Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data, [[maybe_unused]] bool device)
: scalar_(shape.empty())
, sliceShape_(shape)
, sliceOffset_(std::vector<size_t>(std::max(shape.size(), static_cast<size_t>(1)), 0))
, data_(std::make_shared<TensorData>())
{
#if CUDA_ENABLED
    data_->device = device;

    data_->hostDataNeedsUpdate   = data_->device;
    data_->deviceDataNeedsUpdate = false;
#else
    data_->device    = false;
#endif

    data_->shape = shape;

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

Tensor::Tensor(const std::initializer_list<size_t> &shape, const std::initializer_list<float> &data, bool device)
: Tensor(std::vector<size_t>(shape), std::vector<float>(data), device)
{
}

Tensor::Tensor(float value, bool device) : Tensor({}, {value}, device) {}

Tensor::~Tensor() {}

size_t Tensor::size(bool slice) const
{
    const std::vector<size_t> &shape = slice ? sliceShape_ : data_->shape;

    if (shape.empty() && !scalar_)
    {
        return 0;
    }

    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

const std::vector<size_t> &Tensor::shape() const { return sliceShape_; }

bool Tensor::empty() const { return size() == 0; }

bool Tensor::scalar() const { return scalar_; }

void Tensor::toHost()
{
#if CUDA_ENABLED
    if (data_->device)
    {
        data_->device = false;

        if (data_->hostDataNeedsUpdate)
        {
            size_t n = size(false);

            if (data_->hostData == nullptr)
            {
                allocateMemory(n, false);
            }

            cudaMemcpy(data_->hostData.get(), data_->deviceData.get(), n * sizeof(float), cudaMemcpyDeviceToHost);

            data_->hostDataNeedsUpdate = false;
        }

        data_->deviceData.reset(nullptr);
    }
#endif
}

void Tensor::toDevice()
{
#if CUDA_ENABLED
    if (!data_->device)
    {
        data_->device = true;

        size_t n = size(false);

        if (data_->deviceData == nullptr)
        {
            allocateMemory(n, false);
        }

        cudaMemcpy(data_->deviceData.get(), data_->hostData.get(), n * sizeof(float), cudaMemcpyHostToDevice);

        data_->deviceDataNeedsUpdate = false;
    }
#endif
}

void Tensor::eval() const
{
    std::vector<const Tensor *> nodes = sortedNodes();

#if CUDA_ENABLED
    bool leafInDevice;

    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it)
    {
        const Tensor *node = *it;

        if (node->data_->children.empty())
        {
            if (leafInDevice = node->data_->device)
            {
                break;
            }
        }
    }

    if (leafInDevice)
    {
        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it)
        {
            Tensor *node = const_cast<Tensor *>(*it);
            node->toDevice();
        }
    }
#endif

    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it)
    {
        const Tensor *node = *it;

#if CUDA_ENABLED
        if (node->data_->deviceDataNeedsUpdate)
        {
            cudaMemcpy(node->data_->deviceData.get(),
                       node->data_->hostData.get(),
                       node->size(false) * sizeof(float),
                       cudaMemcpyHostToDevice);

            node->data_->deviceDataNeedsUpdate = false;
        }
#endif

        if (node->forward_ != nullptr)
        {
            node->forward_();
        }

#if CUDA_ENABLED
        if (node->data_->device)
        {
            node->data_->hostDataNeedsUpdate = true;
        }
#endif
    }
}

void Tensor::toString(const std::vector<size_t> &indices, std::string &str) const
{
    if (indices.size() == sliceShape_.size())
    {
        size_t idx = index(indices);
        str += std::to_string(data_->hostData[idx]);

        if (indices.back() != sliceShape_.back() - 1)
        {
            str += ", ";
        }
    }
    else
    {
        str += "[";

        for (size_t i = 0; i < sliceShape_[indices.size()]; ++i)
        {
            std::vector<size_t> indices_ = indices;
            indices_.push_back(i);

            toString(indices_, str);
        }

        str += "]";

        if (indices.back() != sliceShape_[indices.size() - 1] - 1)
        {
            std::string lines(sliceShape_.size() - indices.size(), '\n');
            std::string spaces(indices.size() + 7, ' ');

            str += ",";
            str += lines;
            str += spaces;
        }
    }
}

std::string Tensor::toString() const
{
    eval();

#if CUDA_ENABLED
    if (data_->hostDataNeedsUpdate)
    {
        size_t n = size(false);

        if (data_->hostData == nullptr)
        {
            data_->hostData = std::make_unique<float[]>(n);
        }

        cudaMemcpy(data_->hostData.get(), data_->deviceData.get(), n * sizeof(float), cudaMemcpyDeviceToHost);

        data_->hostDataNeedsUpdate = false;
    }
#endif

    std::string str = "Tensor(";

    if (scalar_)
    {
        std::vector<size_t> indices(data_->shape.size(), 0);
        size_t idx = index(indices);

        str += std::to_string(data_->hostData[idx]);
    }
    else
    {
        str += "[";

        if (size() != 0)
        {
            for (size_t i = 0; i < sliceShape_.front(); ++i)
            {
                std::vector<size_t> indices = {i};
                toString(indices, str);
            }
        }

        str += "]";
    }

    str += ", shape=(";

    if (!sliceShape_.empty())
    {
        for (size_t i = 0; i < sliceShape_.size() - 1; ++i)
        {
            str += std::to_string(sliceShape_[i]) + ", ";
        }

        str += std::to_string(sliceShape_.back());
    }

    str += ")";

    if (data_->device)
    {
        str += ", location=device)";
    }
    else
    {
        str += ", location=host)";
    }

    return str;
}

Tensor Tensor::operator+(Tensor &other)
{
    if (other.sliceShape_ != sliceShape_)
    {
        throw std::invalid_argument("Tensor shapes must match for elementwise addition");
    }

    toDevice();
    other.toDevice();

    size_t n = size();

    // TODO: Maybe we should do lazy allocation too
    allocateMemory(n * n, true);
    other.allocateMemory(n * n, true);

    // Set tensor as not realized
    Tensor result(sliceShape_, true);

    result.data_->children.push_back(*this);
    result.data_->children.push_back(other);

    this->data_->parents.push_back(result);
    other.data_->parents.push_back(result);

#if CUDA_ENABLED
    if (result.data_->device)
    {
        result.forward_  = [result, *this, other]() { kernelAddForward(result, *this, other); };
        result.backward_ = [result, *this, other]() { kernelAddBackward(*this, other); };
    }
    else
    {
        result.forward_  = [result, *this, other]() { addForward(result, *this, other); };
        result.backward_ = [result, *this, other]() { addBackward(*this, other); };
    }
#else
    result.forward_  = [result, *this, other]() { addForward(result, *this, other); };
    result.backward_ = [result, *this, other]() { addBackward(*this, other); };
#endif

    return result;
}

Tensor Tensor::operator*(Tensor &other)
{
    if (other.sliceShape_ != sliceShape_)
    {
        throw std::invalid_argument("Tensor shapes must match for elementwise multiplication");
    }

    toDevice();
    other.toDevice();

    size_t n = size();

    // TODO: Maybe we should do lazy allocation too
    allocateMemory(n * n, true);
    other.allocateMemory(n * n, true);

    Tensor result(sliceShape_, true);

    result.data_->children.push_back(*this);
    result.data_->children.push_back(other);

    this->data_->parents.push_back(result);
    other.data_->parents.push_back(result);

#if CUDA_ENABLED
    if (result.data_->device)
    {
        result.forward_  = [result, *this, other]() { kernelMulForward(result, *this, other); };
        result.backward_ = [result, *this, other]() { kernelMulBackward(*this, other); };
    }
    else
    {
        result.forward_  = [result, *this, other]() { mulForward(result, *this, other); };
        result.backward_ = [result, *this, other]() { mulBackward(*this, other); };
    }
#else
    result.forward_  = [result, *this, other]() { mulForward(result, *this, other); };
    result.backward_ = [result, *this, other]() { mulBackward(*this, other); };
#endif

    return result;
}

template<typename... Args> Tensor Tensor::sum(Args... indices) const
{
    // TODO
    return Tensor {};
}

Tensor Tensor::slice(const std::vector<std::vector<size_t>> &ranges) const
{
    for (const std::vector<size_t> &range: ranges)
    {
        if (range.size() != 3)
        {
            throw std::invalid_argument("Slice must be (dimension index, lower bound, upper bound)");
        }
    }

    std::vector<size_t> shape  = sliceShape_;
    std::vector<size_t> offset = sliceOffset_;

    // TODO: Check for repeated indices and ignore them (e.g. reverse sort and use the last)
    for (const std::vector<size_t> &range: ranges)
    {
        const size_t idx = range[0];

        if (idx < sliceShape_.size())
        {
            const int length = std::min(range[2], sliceShape_[idx]) - static_cast<int>(range[1]);
            shape[idx]       = std::max(length, 0);

            offset[idx] += range[1];
        }
    }

    Tensor tensor = *this;

    tensor.sliceShape_  = shape;
    tensor.sliceOffset_ = offset;

    return tensor;
}

size_t Tensor::index(const std::vector<size_t> &indices) const
{
    size_t n = indices.size();

    if (n != data_->shape.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensionality");
    }

    size_t idx    = 0;
    size_t stride = 1;

    for (int i = n - 1; i >= 0; --i)
    {
        if (sliceOffset_[i] + indices[i] >= data_->shape[i])
        {
            throw std::out_of_range("Index out of range");
        }

        // Row-major order
        idx += (sliceOffset_[i] + indices[i]) * stride;
        stride *= data_->shape[i];
    }

    return idx;
}

void Tensor::allocateMemory(size_t size, bool grad) const
{
#if CUDA_ENABLED
    if (data_->device)
    {
        float *tmp;
        cudaMalloc(reinterpret_cast<void **>(&tmp), size * sizeof(float));

        if (grad)
        {
            data_->deviceGrad = std::unique_ptr<float[], TensorData::CudaDeleter>(tmp, TensorData::CudaDeleter {});
        }
        else
        {
            data_->deviceData = std::unique_ptr<float[], TensorData::CudaDeleter>(tmp, TensorData::CudaDeleter {});
        }
    }
    else
    {
        if (grad)
        {
            data_->hostGrad = std::make_unique<float[]>(size);
        }
        else
        {
            data_->hostData = std::make_unique<float[]>(size);
        }
    }
#else
    if (grad)
    {
        data_->hostGrad = std::make_unique<float[]>(size);
    }
    else
    {
        data_->hostData = std::make_unique<float[]>(size);
    }
#endif
}

void Tensor::setData(const std::vector<float> &data)
{
#if CUDA_ENABLED
    if (data_->device)
    {
        cudaMemcpy(data_->deviceData.get(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    else
    {
        std::copy(data.begin(), data.end(), data_->hostData.get());
    }
#else
    std::copy(data.begin(), data.end(), data_->hostData.get());
#endif
}

std::vector<const Tensor *> Tensor::sortedNodes() const
{
    std::unordered_map<const Tensor *, size_t> numParents;

    std::stack<const Tensor *> stack({this});

    // DFS to count the parents in the current tree while ignoring the others
    while (!stack.empty())
    {
        const Tensor *node = stack.top();
        stack.pop();

        for (const Tensor &child: node->data_->children)
        {
            auto search = numParents.find(&child);

            if (search != numParents.end())
            {
                ++numParents[&child];
            }
            else
            {
                numParents[&child] = 1;
                stack.push(&child);
            }
        }
    }

    std::vector<const Tensor *> nodes;

    stack.push(this);

    // Topological sort
    while (!stack.empty())
    {
        const Tensor *node = stack.top();
        stack.pop();

        nodes.push_back(node);

        for (const Tensor &child: node->data_->children)
        {
            --numParents[&child];

            if (numParents[&child] == 0)
            {
                stack.push(&child);
            }
        }
    }

    return nodes;
}

void addForward(const Tensor &result, const Tensor &x, const Tensor &y)
{
    for (size_t i = 0; i < result.size(); ++i)
    {
        result.data_->hostData[i] = x.data_->hostData[i] + y.data_->hostData[i];
    }
}

void addBackward(const Tensor &x, const Tensor &y)
{
    // TODO
    x(0);
    y(0);
}

void mulForward(const Tensor &result, const Tensor &x, const Tensor &y)
{
    for (size_t i = 0; i < result.size(); ++i)
    {
        result.data_->hostData[i] = x.data_->hostData[i] * y.data_->hostData[i];
    }
}

void mulBackward(const Tensor &x, const Tensor &y)
{
    // TODO
    x(0);
    y(0);
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
{
    os << tensor.toString();
    return os;
}
}  // namespace cascade
