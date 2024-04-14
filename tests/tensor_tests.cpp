#include "tensor.h"

#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <vector>

TEST(TensorTests, sum)
{
    constexpr int n = 1e9;

    std::vector<float> x_(n);
    std::iota(x_.begin(), x_.end(), 0.0);

    const std::vector<float> y_(n, 0.5);

    cascade::Tensor x({n}, x_, false);
    cascade::Tensor y({n}, y_, false);

    // "Lazy" elementwise sum (automatically uses GPU if available)
    cascade::Tensor result = x + y;

    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    result.backward_();
}
