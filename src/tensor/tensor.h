#ifndef CASCADE_TENSOR_H
#define CASCADE_TENSOR_H

#include <cstddef>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#if CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace cascade
{
class Tensor
{
public:
    explicit Tensor();

    explicit Tensor(const std::vector<size_t> &shape, bool device = false);
    explicit Tensor(const std::initializer_list<size_t> &shape, bool device = false);

    explicit Tensor(const std::vector<size_t> &shape, const std::vector<float> &data, bool device = false);
    explicit Tensor(const std::initializer_list<size_t> &shape,
                    const std::initializer_list<float> &data,
                    bool device = false);

    explicit Tensor(float value, bool device = false);

    ~Tensor();

    size_t size(bool slice = true) const;
    const std::vector<size_t> &shape() const;

    bool empty() const;
    bool scalar() const;

    template<typename... T>
    Tensor slice(const std::initializer_list<size_t> &firstRange, const std::initializer_list<T> &...otherRanges) const;

    template<typename... T> Tensor operator()(size_t firstIndex, T... otherIndices) const;

    void toHost();
    void toDevice();

    void eval() const;

    std::string toString() const;

    Tensor operator+(Tensor &other);
    Tensor operator-(Tensor &other);
    Tensor operator*(Tensor &other);
    Tensor operator/(Tensor &other);

    template<typename... Args> Tensor sum(Args... indices) const;

    friend void addForward(const Tensor &result, const Tensor &x, const Tensor &y);
    friend void addBackward(const Tensor &x, const Tensor &y);

    friend void kernelAddForward(const Tensor &result, const Tensor &x, const Tensor &y);
    friend void kernelAddBackward(const Tensor &x, const Tensor &y);

    friend void mulForward(const Tensor &result, const Tensor &x, const Tensor &y);
    friend void mulBackward(const Tensor &x, const Tensor &y);

    friend void kernelMulForward(const Tensor &result, const Tensor &x, const Tensor &y);
    friend void kernelMulBackward(const Tensor &x, const Tensor &y);

    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

private:
    Tensor slice(const std::vector<std::vector<size_t>> &ranges) const;

    void toString(const std::vector<size_t> &indices, std::string &str) const;

    size_t index(const std::vector<size_t> &indices) const;

    void allocateMemory(size_t size, bool grad) const;

    void setData(const std::vector<float> &data);

    std::vector<const Tensor *> sortedNodes() const;

    bool scalar_;

    std::vector<size_t> sliceShape_;
    std::vector<size_t> sliceOffset_;

    std::function<void()> forward_;
    std::function<void()> backward_;

    struct TensorData;
    std::shared_ptr<TensorData> data_;
};
}  // namespace cascade

#include "tensor.inl.h"

#endif
