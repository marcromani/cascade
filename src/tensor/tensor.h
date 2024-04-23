#ifndef CASCADE_TENSOR_H
#define CASCADE_TENSOR_H

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#if CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace cascade
{
class Tensor
{
public:
    explicit Tensor(bool device = false);
    explicit Tensor(float value, bool device = false);
    explicit Tensor(const std::vector<size_t> &shape, bool device = false);
    explicit Tensor(const std::vector<size_t> &shape, const std::vector<float> &data, bool device = false);

    ~Tensor();

    size_t size() const;
    const std::vector<size_t> &shape() const;

    template<typename... Args> float operator[](Args... indices) const;

    class ProxyValue;
    template<typename... Args> ProxyValue operator[](Args... indices);

    void toHost();
    void toDevice();

    Tensor operator+(Tensor &other);
    Tensor operator-(Tensor &other);
    Tensor operator*(Tensor &other);
    Tensor operator/(Tensor &other);

    template<typename... Args> Tensor sum(Args... indices) const;

    void eval() const;

    friend void addForward(const Tensor &result, const Tensor &x, const Tensor &y);
    friend void addBackward(const Tensor &x, const Tensor &y);

    friend void kernelAddForward(const Tensor &result, const Tensor &x, const Tensor &y);
    friend void kernelAddBackward(const Tensor &x, const Tensor &y);

    friend void mulForward(const Tensor &result, const Tensor &x, const Tensor &y);
    friend void mulBackward(const Tensor &x, const Tensor &y);

    friend void kernelMulForward(const Tensor &result, const Tensor &x, const Tensor &y);
    friend void kernelMulBackward(const Tensor &x, const Tensor &y);

private:
    struct CudaDeleter final
    {
#if CUDA_ENABLED
        void operator()(float *ptr) const { cudaFree(ptr); }
#else
        void operator()(float *ptr) const { delete[] ptr; }
#endif
    };

    size_t index(const std::vector<size_t> &indices) const;

    void allocateMemory(size_t size, bool grad);

    void setData(const std::vector<float> &data);

    std::vector<const Tensor *> sortedNodes() const;

    std::vector<size_t> shape_;

    std::vector<std::shared_ptr<Tensor>> children_;
    std::vector<std::shared_ptr<Tensor>> parents_;

public:
    std::function<void()> forward_;
    std::function<void()> backward_;

    class TensorData;
    std::shared_ptr<TensorData> data_;
};
}  // namespace cascade

#include "tensor.h.inl"

#endif
