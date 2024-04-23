#ifndef CASCADE_TENSOR_H_INL
#define CASCADE_TENSOR_H_INL

#ifndef CASCADE_TENSOR_H
#error __FILE__ should only be included from tensor.h
#endif

#include "proxy_value.h"
#include "tensor_data.h"

#include <cstddef>
#include <memory>
#include <type_traits>

#if CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace cascade
{
template<typename... Args> float Tensor::operator[](Args... indices) const
{
    static_assert(std::conjunction_v<std::disjunction<std::is_same<Args, size_t>, std::is_same<Args, int>>...>,
                  "Indices must be of type size_t or int");

    size_t idx = index({static_cast<size_t>(indices)...});

    eval();  // TODO: Not always

#if CUDA_ENABLED
    if (data_->hostDataNeedsUpdate)
    {
        size_t n = size();

        if (data_->hostData == nullptr)
        {
            data_->hostData = std::make_unique<float[]>(n);
        }

        cudaMemcpy(data_->hostData.get(), data_->deviceData.get(), n * sizeof(float), cudaMemcpyDeviceToHost);

        data_->hostDataNeedsUpdate = false;
    }
#endif

    return data_->hostData[idx];
}

template<typename... Args> Tensor::ProxyValue Tensor::operator[](Args... indices)
{
    static_assert(std::conjunction_v<std::disjunction<std::is_same<Args, size_t>, std::is_same<Args, int>>...>,
                  "Indices must be of type size_t or int");

    size_t idx = index({static_cast<size_t>(indices)...});

    eval();  // TODO: Not always

#if CUDA_ENABLED
    if (data_->device)
    {
        if (data_->hostDataNeedsUpdate)
        {
            size_t n = size();

            if (data_->hostData == nullptr)
            {
                data_->hostData = std::make_unique<float[]>(n);
            }

            cudaMemcpy(data_->hostData.get(), data_->deviceData.get(), n * sizeof(float), cudaMemcpyDeviceToHost);

            data_->hostDataNeedsUpdate = false;
        }

        return ProxyValue(data_->hostData[idx], &data_->deviceDataNeedsUpdate);
    }
    else
    {
        return ProxyValue(data_->hostData[idx], nullptr);
    }
#else
    return ProxyValue(data_->hostData[idx], nullptr);
#endif
}
}  // namespace cascade

#endif
