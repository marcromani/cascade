#include "tensor.h"

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

TEST(TensorTests, sum)
{
    std::vector<float> x_ = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> y_ = {0.5, 0.5, 0.5, 0.5};

    Tensor x({4}, x_);
    Tensor y({4}, y_);

    // Elementwise sum (automatically uses GPU if available)
    Tensor result = x + y;

    for (size_t i = 0; i < result.size(); ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
}
