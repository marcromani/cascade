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
class Tensor final
{
public:
    explicit Tensor(bool device = false);
    explicit Tensor(float value, bool device = false);
    explicit Tensor(const std::vector<size_t> &shape, bool device = false);
    explicit Tensor(const std::vector<size_t> &shape, const std::vector<float> &data, bool device = false);

    ~Tensor();

    size_t size() const;
    const std::vector<size_t> &shape() const;

    template<typename... Args> const float &operator[](Args... indices) const;
    template<typename... Args> float &operator[](Args... indices);

    Tensor toHost() const;
    Tensor toDevice() const;

    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;
    Tensor operator/(const Tensor &other) const;

    template<typename... Args> Tensor sum(Args... indices) const;

private:
    size_t index(const std::vector<size_t> &indices) const;

    void allocateMemory(size_t size, bool grad);

    void setData(const std::vector<float> &data);

public:
    bool device_;

    bool hostDataNeedsUpdate_;
    bool deviceDataNeedsUpdate_;

    std::vector<size_t> shape_;

    mutable std::shared_ptr<float[]> hostData_;
    mutable std::shared_ptr<float[]> hostGrad_;

    std::shared_ptr<float[]> deviceData_;
    std::shared_ptr<float[]> deviceGrad_;

    std::vector<Tensor> children_;
    std::vector<Tensor> parents_;

    std::function<void()> forward_;
    std::function<void()> backward_;
};
}  // namespace cascade

#include "tensor.h.inl"

#endif
