#ifndef CASCADE_TENSOR_TPP
#define CASCADE_TENSOR_TPP

#ifndef CASCADE_TENSOR_H
#error __FILE__ should only be included from tensor.h
#endif

#include <memory>
#include <type_traits>

#if CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace cascade
{
template<typename... Args> const float &Tensor::operator[](Args... indices) const
{
    static_assert(std::conjunction_v<std::is_same<Args, size_t>...>, "Indices must be of type size_t");

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
}  // namespace cascade

#endif
