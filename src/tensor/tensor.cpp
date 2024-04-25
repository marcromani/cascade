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
Tensor::Tensor([[maybe_unused]] bool device) : data_(std::make_shared<TensorData>())
{
#if CUDA_ENABLED
    data_->device = device;
#else
    data_->device = false;
#endif

    data_->hostDataNeedsUpdate   = false;
    data_->deviceDataNeedsUpdate = false;
}

Tensor::Tensor(float value, bool device) : Tensor({1}, {value}, device) {}

Tensor::Tensor(const std::vector<size_t> &shape, [[maybe_unused]] bool device)
: shape_(shape)
, offset_({shape.size(), 0})
, data_(std::make_shared<TensorData>())
{
#if CUDA_ENABLED
    data_->device = device;

    data_->hostDataNeedsUpdate   = data_->device;
    data_->deviceDataNeedsUpdate = !data_->device;
#else
    data_->device = false;

    data_->hostDataNeedsUpdate   = false;
    data_->deviceDataNeedsUpdate = false;
#endif

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
            std::fill(data_->hostData.get(), data_->hostData.get() + n, 0.f);
        }
    }
}

Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data, [[maybe_unused]] bool device)
: shape_(shape)
, offset_({shape.size(), 0})
, data_(std::make_shared<TensorData>())
{
#if CUDA_ENABLED
    data_->device = device;

    data_->hostDataNeedsUpdate   = data_->device;
    data_->deviceDataNeedsUpdate = !data_->device;
#else
    data_->device                = false;

    data_->hostDataNeedsUpdate   = false;
    data_->deviceDataNeedsUpdate = false;
#endif

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

void Tensor::toHost()
{
#if CUDA_ENABLED
    if (data_->device)
    {
        data_->device = false;

        if (data_->hostDataNeedsUpdate)
        {
            size_t n = size();

            if (data_->hostData == nullptr)
            {
                allocateMemory(n, false);
            }

            cudaMemcpy(data_->hostData.get(), data_->deviceData.get(), n * sizeof(float), cudaMemcpyDeviceToHost);

            data_->hostDataNeedsUpdate   = false;
            data_->deviceDataNeedsUpdate = true;
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

        if (data_->deviceDataNeedsUpdate)
        {
            size_t n = size();

            if (data_->deviceData == nullptr)
            {
                allocateMemory(n, false);
            }

            cudaMemcpy(data_->deviceData.get(), data_->hostData.get(), n * sizeof(float), cudaMemcpyHostToDevice);

            data_->hostDataNeedsUpdate   = true;
            data_->deviceDataNeedsUpdate = false;
        }

        data_->hostData.reset(nullptr);
    }
#endif
}

void Tensor::eval() const
{
    std::vector<const Tensor *> nodes = sortedNodes();

    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it)
    {
        const Tensor *node = *it;

        if (node->forward_ != nullptr)
        {
            node->forward_();

#if CUDA_ENABLED
            if (node->data_->device)
            {
                node->data_->hostDataNeedsUpdate = true;
            }
            else
            {
                node->data_->deviceDataNeedsUpdate = true;
            }
#endif
        }
    }
}

void Tensor::toString(const std::vector<size_t> &indices, std::string &str) const
{
    if (indices.size() == shape_.size())
    {
        size_t idx = index(indices);
        str += std::to_string(idx);

        if (indices.back() != shape_.back() - 1)
        {
            str += ", ";
        }
    }
    else
    {
        str += "[";

        for (int i = 0; i < shape_[indices.size()]; ++i)
        {
            std::vector<size_t> indices_ = indices;
            indices_.push_back(i);

            toString(indices_, str);
        }

        str += "]";

        if (indices.back() != shape_[indices.size() - 1] - 1)
        {
            std::string spaces(indices.size(), ' ');
            str += ",";
            str += "\n";
            str += spaces;
        }
    }
}

std::string Tensor::toString() const
{
    std::string str = "[";

    for (size_t i = 0; i < shape_.front(); ++i)
    {
        std::vector<size_t> indices = {i};
        toString(indices, str);
    }

    str += "]";

    return str;
}

Tensor Tensor::operator+(Tensor &other)
{
    if (other.shape_ != shape_)
    {
        throw std::invalid_argument("Tensor shapes must match for elementwise addition");
    }

    toDevice();
    other.toDevice();

    size_t n = size();

    // TODO: Maybe we should do lazy allocation too
    allocateMemory(n * n, true);
    other.allocateMemory(n * n, true);

    Tensor result(shape_, true);

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
    result.forward_              = [result, *this, other]() { addForward(result, *this, other); };
    result.backward_             = [result, *this, other]() { addBackward(*this, other); };
#endif

    return result;
}

Tensor Tensor::operator*(Tensor &other)
{
    if (other.shape_ != shape_)
    {
        throw std::invalid_argument("Tensor shapes must match for elementwise multiplication");
    }

    toDevice();
    other.toDevice();

    size_t n = size();

    // TODO: Maybe we should do lazy allocation too
    allocateMemory(n * n, true);
    other.allocateMemory(n * n, true);

    Tensor result(shape_, true);

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
    result.forward_              = [result, *this, other]() { mulForward(result, *this, other); };
    result.backward_             = [result, *this, other]() { mulBackward(*this, other); };
#endif

    return result;
}

template<typename... Args> Tensor Tensor::sum(Args... indices) const
{
    // TODO
    return Tensor {};
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
    // x[0];
    // y[0];
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
    // x[0];
    // y[0];
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
{
    os << tensor.toString();
    return os;
}
}  // namespace cascade
