#include "tensor.h"

#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <vector>

TEST(TensorTests, sum)
{
    constexpr int n = 1e3;

    std::vector<float> x_(n);
    std::iota(x_.begin(), x_.end(), 0.f);

    const std::vector<float> y_(n, 0.5f);

    cascade::Tensor x({n}, x_, false);
    cascade::Tensor y({n}, y_, false);

    // "Lazy" elementwise sum (automatically uses GPU if available)
    cascade::Tensor result = x + y;

    result.forward_();

    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    result.backward_();

    size_t size = result.size() * result.size();

    float *grad = new float[size];
    cudaMemcpy(grad, x.deviceGrad_.get(), size * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << grad[i] << " ";
    }
    std::cout << std::endl;

    delete[] grad;
}
