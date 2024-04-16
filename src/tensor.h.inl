#ifndef CASCADE_TENSOR_H_INL
#define CASCADE_TENSOR_H_INL

#ifndef CASCADE_TENSOR_H
#error __FILE__ should only be included from tensor.h
#endif

#include <cstddef>
#include <memory>
#include <type_traits>

#if CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace cascade
{
template<typename... Args> const float &Tensor::operator[](Args... indices) const
{
    static_assert(std::conjunction_v<std::disjunction<std::is_same<Args, size_t>, std::is_same<Args, int>>...>,
                  "Indices must be of type size_t or int");

    size_t idx = index({static_cast<size_t>(indices)...});

#if CUDA_ENABLED
    if (hostData_ == nullptr)
    {
        size_t n = size();

        hostData_ = std::shared_ptr<float[]>(new float[n]);
        cudaMemcpy(hostData_.get(), deviceData_.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    }
#endif

    return hostData_[idx];
}
}  // namespace cascade

#endif
