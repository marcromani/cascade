#include "tensor.h"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

TEST(TensorTests, emptyTensorHasCorrectPropertiesAfterInitialization)
{
    cascade::Tensor tensor;

    EXPECT_TRUE(tensor.empty()) << "Empty tensor reports it is not empty";
    EXPECT_EQ(tensor.size(), 0) << "Empty tensor reports its size is not 0";
    EXPECT_TRUE(tensor.shape() == std::vector<size_t> {}) << "Empty tensor shape is not empty";
    EXPECT_FALSE(tensor.scalar()) << "Empty tensor reports it is a scalar";

    tensor = cascade::Tensor({});

    EXPECT_TRUE(tensor.empty()) << "Empty tensor reports it is not empty";
    EXPECT_EQ(tensor.size(), 0) << "Empty tensor reports its size is not 0";
    EXPECT_TRUE(tensor.shape() == std::vector<size_t> {}) << "Empty tensor shape is not empty";
    EXPECT_FALSE(tensor.scalar()) << "Empty tensor reports it is a scalar";
}

TEST(TensorTests, scalarTensorHasCorrectPropertiesAfterInitialization)
{
    cascade::Tensor tensor(12.5f);

    EXPECT_FALSE(tensor.empty()) << "Scalar tensor reports it is empty";
    EXPECT_EQ(tensor.size(), 1) << "Scalar tensor reports its size is not 1";
    EXPECT_TRUE(tensor.shape() == std::vector<size_t> {}) << "Scalar tensor shape is not empty";
    EXPECT_TRUE(tensor.scalar()) << "Scalar tensor reports it is not a scalar";
}

TEST(TensorTests, higherOrderTensorHasCorrectPropertiesAfterInitialization)
{
    cascade::Tensor tensor({3, 5, 1, 2});

    EXPECT_FALSE(tensor.empty()) << "Tensor reports it is empty";
    EXPECT_EQ(tensor.size(), 30) << "Tensor reports wrong size";
    EXPECT_TRUE(tensor.shape() == (std::vector<size_t> {3, 5, 1, 2})) << "Tensor has wrong shape";
    EXPECT_FALSE(tensor.scalar()) << "Tensor reports it is a scalar";
}

TEST(TensorTests, degenerateHigherOrderTensorHasCorrectPropertiesAfterInitialization)
{
    cascade::Tensor tensor({1, 0, 7, 3});

    EXPECT_TRUE(tensor.empty()) << "Tensor reports it is not empty";
    EXPECT_EQ(tensor.size(), 0) << "Tensor reports wrong size";
    EXPECT_TRUE(tensor.shape() == (std::vector<size_t> {1, 0, 7, 3})) << "Tensor has wrong shape";
    EXPECT_FALSE(tensor.scalar()) << "Tensor reports it is a scalar";
}

TEST(TensorTests, appropriateConstructorOverloadIsChosen)
{
    EXPECT_THROW(cascade::Tensor({}, {static_cast<int>(2.5f)}), std::invalid_argument)
        << "Exception not thrown, constructor from data not called or it does not throw";
}

TEST(TensorTests, constructorFromDataThrows)
{
    EXPECT_THROW(cascade::Tensor({4, 3, 5, 1}, {1.1f, 4.6f, 3.7f}), std::invalid_argument)
        << "Constructor from data does not throw";
}

TEST(TensorTests, sum)
{
    constexpr size_t m = 2;
    constexpr size_t n = 3;
    constexpr size_t r = 2;

    // std::vector<float> x_(n);
    // std::iota(x_.begin(), x_.end(), 0.f);

    // const std::vector<float> y_(n, 0.5f);

    cascade::Tensor x({m, n, r}, false);
    cascade::Tensor y({m, n, r}, false);

    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            for (size_t k = 0; k < r; ++k)
            {
                x(i, j, k) = i + j + k;
                y(i, j, k) = 0.5 * j;
            }
        }
    }

    // Lazy elementwise multiplication (automatically uses GPU if available)
    cascade::Tensor z = x * y;

    cascade::Tensor w({m, n, r}, false);

    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            for (size_t k = 0; k < r; ++k)
            {
                w(i, j, k) = 1.1f;
            }
        }
    }

    // Lazy elementwise sum (automatically uses GPU if available)
    w = w + z;

    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            for (size_t k = 0; k < r; ++k)
            {
                std::cout << w(i, j, k) << " ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << w << std::endl;

    cascade::Tensor a({2, 3, 2, 4, 3}, false);
    std::cout << a << std::endl;

    cascade::Tensor b({1, 1, 1, 1}, false);
    std::cout << b << std::endl;

    // cascade::Tensor w1({50, 10});
    // cascade::Tensor w2({100, 50});
    // cascade::Tensor w3({25, 100});
    // cascade::Tensor w4({1, 25});

    // cascade::Tensor x({10, 1000});
    // cascade::Tensor y({1, 1000});

    // cascade::Tensor t = cascade::relu(cascade::matmul(w1, x));
    // t = cascade::relu(cascade::matmul(w2, t));
    // t = cascade::relu(cascade::matmul(w3, t));
    // t = cascade::relu(cascade::matmul(w4, t));

    // cascade::Tensor loss = cascade::sum(cascade::pow(t - y, 2));

    // result.backward_();

    // size_t size = result.size() * result.size();

    // // TODO: Should copy the gradients too
    // x.toHost();

    // for (size_t i = 0; i < size; ++i)
    // {
    //     std::cout << x.data_->hostGrad[i] << " ";
    // }
    // std::cout << std::endl;
}
