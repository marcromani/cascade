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
}

TEST(TensorTests, scalarTensorHasCorrectPropertiesAfterInitialization)
{
    cascade::Tensor tensor({}, false);

    EXPECT_FALSE(tensor.empty()) << "Scalar tensor reports it is empty";
    EXPECT_EQ(tensor.size(), 1) << "Scalar tensor reports its size is not 1";
    EXPECT_TRUE(tensor.shape() == std::vector<size_t> {}) << "Scalar tensor shape is not empty";
    EXPECT_TRUE(tensor.scalar()) << "Scalar tensor reports it is not a scalar";

    tensor = cascade::Tensor({}, true);

    EXPECT_FALSE(tensor.empty()) << "Scalar tensor reports it is empty";
    EXPECT_EQ(tensor.size(), 1) << "Scalar tensor reports its size is not 1";
    EXPECT_TRUE(tensor.shape() == std::vector<size_t> {}) << "Scalar tensor shape is not empty";
    EXPECT_TRUE(tensor.scalar()) << "Scalar tensor reports it is not a scalar";

    tensor = cascade::Tensor(12.5f, false);

    EXPECT_FALSE(tensor.empty()) << "Scalar tensor reports it is empty";
    EXPECT_EQ(tensor.size(), 1) << "Scalar tensor reports its size is not 1";
    EXPECT_TRUE(tensor.shape() == std::vector<size_t> {}) << "Scalar tensor shape is not empty";
    EXPECT_TRUE(tensor.scalar()) << "Scalar tensor reports it is not a scalar";

    tensor = cascade::Tensor(12.5f, true);

    EXPECT_FALSE(tensor.empty()) << "Scalar tensor reports it is empty";
    EXPECT_EQ(tensor.size(), 1) << "Scalar tensor reports its size is not 1";
    EXPECT_TRUE(tensor.shape() == std::vector<size_t> {}) << "Scalar tensor shape is not empty";
    EXPECT_TRUE(tensor.scalar()) << "Scalar tensor reports it is not a scalar";
}

TEST(TensorTests, higherOrderTensorHasCorrectPropertiesAfterInitialization)
{
    cascade::Tensor tensor({3, 5, 1, 2}, false);

    EXPECT_FALSE(tensor.empty()) << "Tensor reports it is empty";
    EXPECT_EQ(tensor.size(), 30) << "Tensor reports wrong size";
    EXPECT_TRUE(tensor.shape() == (std::vector<size_t> {3, 5, 1, 2})) << "Tensor has wrong shape";
    EXPECT_FALSE(tensor.scalar()) << "Tensor reports it is a scalar";

    tensor = cascade::Tensor({3, 5, 1, 2}, true);

    EXPECT_FALSE(tensor.empty()) << "Tensor reports it is empty";
    EXPECT_EQ(tensor.size(), 30) << "Tensor reports wrong size";
    EXPECT_TRUE(tensor.shape() == (std::vector<size_t> {3, 5, 1, 2})) << "Tensor has wrong shape";
    EXPECT_FALSE(tensor.scalar()) << "Tensor reports it is a scalar";
}

TEST(TensorTests, degenerateTensorHasCorrectPropertiesAfterInitialization)
{
    cascade::Tensor tensor({0});

    EXPECT_TRUE(tensor.empty()) << "Tensor reports it is not empty";
    EXPECT_EQ(tensor.size(), 0) << "Tensor reports wrong size";
    EXPECT_TRUE(tensor.shape() == (std::vector<size_t> {0})) << "Tensor has wrong shape";
    EXPECT_FALSE(tensor.scalar()) << "Tensor reports it is not a scalar";

    tensor = cascade::Tensor({1, 0, 7, 3});

    EXPECT_TRUE(tensor.empty()) << "Tensor reports it is not empty";
    EXPECT_EQ(tensor.size(), 0) << "Tensor reports wrong size";
    EXPECT_TRUE(tensor.shape() == (std::vector<size_t> {1, 0, 7, 3})) << "Tensor has wrong shape";
    EXPECT_FALSE(tensor.scalar()) << "Tensor reports it is a scalar";
}

TEST(TensorTests, appropriateConstructorOverloadIsChosen)
{
    cascade::Tensor tensor({}, {static_cast<int>(2.5f)});

    // TODO
    // EXPECT_FLOAT_EQ(tensor.data_->hostData.get()[0], 2.f);
}

TEST(TensorTests, constructorFromDataThrows)
{
    EXPECT_THROW(cascade::Tensor({4, 3, 5, 1}, {1.1f, 4.6f, 3.7f}), std::invalid_argument)
        << "Constructor from data does not throw";
}

TEST(TensorTests, sliceInsideBounds)
{
    cascade::Tensor tensor({3, 4, 1, 2});

    cascade::Tensor slice = tensor.slice({0, 1, 2}, {1, 2, 4});

    EXPECT_FALSE(slice.empty()) << "Tensor slice reports it is empty";
    EXPECT_EQ(slice.size(), 4) << "Tensor slice reports wrong size";
    EXPECT_TRUE(slice.shape() == (std::vector<size_t> {1, 2, 1, 2})) << "Tensor slice has wrong shape";
    EXPECT_FALSE(slice.scalar()) << "Tensor reports it is a scalar";

    slice = tensor.slice({0, 1, 2}, {1, 4, 2});

    EXPECT_TRUE(slice.empty()) << "Tensor slice reports it is not empty";
    EXPECT_EQ(slice.size(), 0) << "Tensor slice reports wrong size";
    EXPECT_TRUE(slice.shape() == (std::vector<size_t> {1, 0, 1, 2})) << "Tensor slice has wrong shape";
    EXPECT_FALSE(slice.scalar()) << "Tensor reports it is a scalar";
}

TEST(TensorTests, sliceOutsideBounds)
{
    cascade::Tensor tensor({2, 5, 7, 3});

    cascade::Tensor slice = tensor.slice({2, 8, 10}, {1, 2, 4});

    EXPECT_TRUE(slice.empty()) << "Tensor slice reports it is not empty";
    EXPECT_EQ(slice.size(), 0) << "Tensor slice reports wrong size";
    EXPECT_TRUE(slice.shape() == (std::vector<size_t> {2, 2, 0, 3})) << "Tensor slice has wrong shape";
    EXPECT_FALSE(slice.scalar()) << "Tensor reports it is a scalar";

    slice = tensor.slice({2, 8, 10}, {1, 4, 2});

    EXPECT_TRUE(slice.empty()) << "Tensor slice reports it is not empty";
    EXPECT_EQ(slice.size(), 0) << "Tensor slice reports wrong size";
    EXPECT_TRUE(slice.shape() == (std::vector<size_t> {2, 0, 0, 3})) << "Tensor slice has wrong shape";
    EXPECT_FALSE(slice.scalar()) << "Tensor reports it is a scalar";
}

TEST(TensorTests, sliceCrossingBounds)
{
    cascade::Tensor tensor({4, 6, 2, 1});

    cascade::Tensor slice = tensor.slice({1, 3, 8}, {0, 2, 3}, {3, 0, 1});

    EXPECT_FALSE(slice.empty()) << "Tensor slice reports it is empty";
    EXPECT_EQ(slice.size(), 6) << "Tensor slice reports wrong size";
    EXPECT_TRUE(slice.shape() == (std::vector<size_t> {1, 3, 2, 1})) << "Tensor slice has wrong shape";
    EXPECT_FALSE(slice.scalar()) << "Tensor reports it is a scalar";

    slice = tensor.slice({1, 8, 3}, {0, 2, 3}, {3, 0, 1});

    EXPECT_TRUE(slice.empty()) << "Tensor slice reports it is not empty";
    EXPECT_EQ(slice.size(), 0) << "Tensor slice reports wrong size";
    EXPECT_TRUE(slice.shape() == (std::vector<size_t> {1, 0, 2, 1})) << "Tensor slice has wrong shape";
    EXPECT_FALSE(slice.scalar()) << "Tensor reports it is a scalar";
}

TEST(TensorTests, slicingWithWrongInitializerListThrows)
{
    cascade::Tensor tensor({1, 2, 3, 4});

    EXPECT_THROW(tensor.slice({}), std::invalid_argument) << "Slicing with wrong arguments does not throw";
    EXPECT_THROW(tensor.slice({3, 1}), std::invalid_argument) << "Slicing with wrong arguments does not throw";
    EXPECT_THROW(tensor.slice({3, 1, 2, 1}), std::invalid_argument) << "Slicing with wrong arguments does not throw";

    EXPECT_NO_THROW(tensor.slice({3, 1, 2})) << "Slicing with correct arguments throws";
}

TEST(TensorTests, slicingEmptyTensorHasNoEffect)
{
    cascade::Tensor tensor;

    cascade::Tensor slice = tensor.slice({3, 0, 5});

    EXPECT_TRUE(slice.empty()) << "Tensor slice reports it is not empty";
    EXPECT_EQ(slice.size(), 0) << "Tensor slice reports wrong size";
    EXPECT_TRUE(slice.shape() == (std::vector<size_t> {})) << "Tensor slice has wrong shape";
    EXPECT_FALSE(slice.scalar()) << "Tensor reports it is a scalar";
}

TEST(TensorTests, slicingScalarTensorThrows)
{
    cascade::Tensor tensor({}, {32.8});

    EXPECT_THROW(tensor.slice({0, 0, 1}), std::invalid_argument) << "Slicing a scalar tensor does not throw";
}

TEST(TensorTests, elementAccessWithWrongNumberOfIndicesThrows)
{
    cascade::Tensor tensor({5, 2, 3, 2});

    EXPECT_THROW(tensor(2), std::invalid_argument) << "Element access with wrong arguments does not throw";
    EXPECT_THROW(tensor(2, 1), std::invalid_argument) << "Element access with wrong arguments does not throw";
    EXPECT_THROW(tensor(2, 1, 2), std::invalid_argument) << "Element access with wrong arguments does not throw";
    EXPECT_THROW(tensor(2, 1, 2, 1, 3), std::invalid_argument) << "Element access with wrong arguments does not throw";

    EXPECT_NO_THROW(tensor(2, 1, 2, 1)) << "Element access with correct arguments throws";
}

TEST(TensorTests, elementAccessEmptyTensorThrows)
{
    cascade::Tensor tensor({5, 0, 3, 2});

    EXPECT_THROW(tensor(4, 1, 2, 1), std::invalid_argument) << "Element access of empty tensor does not throw";
}

TEST(TensorTests, elementAccess)
{
    std::vector<float> data(120);
    std::iota(data.begin(), data.end(), 0.f);

    cascade::Tensor tensor({5, 3, 4, 2}, data);

    cascade::Tensor scalar = tensor(2, 1, 3, 0);

    // scalar.item() == 62.f;
}

TEST(TensorTests, sum)
{
    // std::cout << "Empty tensor:" << std::endl;
    // std::cout << cascade::Tensor() << std::endl;

    // std::cout << "\nUninitialized tensors:" << std::endl;
    // std::cout << cascade::Tensor({}) << std::endl;         // Scalar
    // std::cout << cascade::Tensor({5}) << std::endl;        // Vector
    // std::cout << cascade::Tensor({2, 3}) << std::endl;     // Matrix
    // std::cout << cascade::Tensor({2, 2, 2}) << std::endl;  // Rank 3 tensor

    // std::cout << "\nTensors initialized with provided data:" << std::endl;
    // std::cout << cascade::Tensor({}, {1.f}) << std::endl;                                            // Scalar
    // std::cout << cascade::Tensor({5}, {1.f, 2.f, 3.f, 4.f, 5.f}) << std::endl;                       // Vector
    // std::cout << cascade::Tensor({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}) << std::endl;               // Matrix
    // std::cout << cascade::Tensor({2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, true)(1, 1, 1) << std::endl;  // Rank 3 tensor

    cascade::Tensor t({}, {23.4});
    // cascade::Tensor t;
    // std::cout << t(0) << std::endl;
    std::cout << t.slice({0, 5, 2}) << std::endl;
    return;

    // std::cout << "\nTensors filled with zeros:" << std::endl;
    // std::cout << cascade::Tensor::zeros({}) << std::endl;         // Scalar
    // std::cout << cascade::Tensor::zeros({5}) << std::endl;        // Vector
    // std::cout << cascade::Tensor::zeros({2, 3}) << std::endl;     // Matrix
    // std::cout << cascade::Tensor::zeros({2, 1, 4}) << std::endl;  // Rank 3 tensor

    // std::cout << "\nTensors filled with ones:" << std::endl;
    // std::cout << cascade::Tensor::ones({}) << std::endl;         // Scalar
    // std::cout << cascade::Tensor::ones({5}) << std::endl;        // Vector
    // std::cout << cascade::Tensor::ones({2, 3}) << std::endl;     // Matrix
    // std::cout << cascade::Tensor::ones({2, 1, 4}) << std::endl;  // Rank 3 tensor

    std::cout << "\nDegenerate tensors:" << std::endl;
    std::cout << cascade::Tensor({0}) << std::endl;        // Vector
    std::cout << cascade::Tensor({0, 3}) << std::endl;     // Matrix
    std::cout << cascade::Tensor({2, 0, 2}) << std::endl;  // Rank 3 tensor

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
                // x(i, j, k) = i + j + k;
                // y(i, j, k) = 0.5 * j;
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
                // w(i, j, k) = 1.1f;
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

    cascade::Tensor scalar(27.4);

    std::cout << scalar << std::endl;
    std::cout << cascade::Tensor() << std::endl;

    std::cout << cascade::Tensor({7}, true) << std::endl;

    std::cout << cascade::Tensor({3, 0, 2, 5}, true) << std::endl;

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
