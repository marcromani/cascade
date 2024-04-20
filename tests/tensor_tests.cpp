#include "tensor.h"

#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <vector>

TEST(TensorTests, sum)
{
    constexpr int n = 5;

    // std::vector<float> x_(n);
    // std::iota(x_.begin(), x_.end(), 0.f);

    // const std::vector<float> y_(n, 0.5f);

    cascade::Tensor x(std::vector<size_t> {n}, false);
    cascade::Tensor y(std::vector<size_t> {n}, false);

    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = i;
        y[i] = 0.5 * i;
    }

    // "Lazy" elementwise sum (automatically uses GPU if available)
    cascade::Tensor result = x * y;

    result.forward_();

    for (size_t i = 0; i < result.size(); ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    result.backward_();

    size_t size = result.size() * result.size();

    // TODO: Should copy the gradients too
    x.toHost();

    for (size_t i = 0; i < size; ++i)
    {
        std::cout << x.data_->hostGrad[i] << " ";
    }
    std::cout << std::endl;
}
