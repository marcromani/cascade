#ifndef CASCADE_TENSOR_H_INL
#define CASCADE_TENSOR_H_INL

#ifndef CASCADE_TENSOR_H
#error __FILE__ should only be included from tensor.h
#endif

#include "tensor_data.h"

#include <cstddef>
#include <type_traits>
#include <vector>

#if CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace cascade
{
template<typename... T>
Tensor Tensor::slice(const std::initializer_list<size_t> &firstRange,
                     const std::initializer_list<T> &...otherRanges) const
{
    static_assert(std::conjunction_v<std::disjunction<std::is_same<T, size_t>, std::is_same<T, int>>...>,
                  "Ranges must be initializer lists of type size_t or int");

    std::vector<std::vector<size_t>> rangesVector = {{firstRange.begin(), firstRange.end()}};
    ((rangesVector.emplace_back(otherRanges.begin(), otherRanges.end())), ...);

    return slice(rangesVector);
}

template<typename... T> Tensor Tensor::operator()(size_t firstIndex, T... otherIndices) const
{
    static_assert(std::conjunction_v<std::disjunction<std::is_same<T, size_t>, std::is_same<T, int>>...>,
                  "Indices must be of type size_t or int");

    std::vector<size_t> indicesVector = {firstIndex};
    ((indicesVector.push_back(static_cast<size_t>(otherIndices))), ...);

    if (empty())
    {
        throw std::invalid_argument("Cannot access element of an empty tensor");
    }

    if (indicesVector.size() != shape_.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensionality");
    }

    std::vector<std::vector<size_t>> rangesVector;

    for (size_t i = 0; i < indicesVector.size(); ++i)
    {
        rangesVector.push_back({i, indicesVector[i], indicesVector[i] + 1});
    }

    Tensor tensor = slice(rangesVector);

    tensor.scalar_ = true;
    tensor.shape_  = {};

    return tensor;
}
}  // namespace cascade

#endif
