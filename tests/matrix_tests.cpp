#include "util.h"

#include <gtest/gtest.h>
#include <vector>

TEST(MatrixTests, rectangularByRectangularProductTest)
{
    const std::vector<double> A {1, 2, 3, 0.1, 0.2, 0.3};
    const std::vector<double> B {1, -1, 1, 0, 1, 4, 1, 0.5, -1};

    const std::vector<double> result = cascade::multiply(A, B, 2);

    const std::vector<double> expected = {4, 2.5, 6, 0.4, 0.25, 0.6};

    ASSERT_EQ(result.size(), expected.size()) << "Matrix product has wrong size";

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(result[i], expected[i]) << "Matrix product has wrong value at index " << i;
    }
}

TEST(MatrixTests, rowByColumnProductTest)
{
    const std::vector<double> A {1, -2, 3, -4, 5};
    const std::vector<double> B {-1, 2, -3, 4, -5};

    const std::vector<double> result = cascade::multiply(A, B, 1);

    const std::vector<double> expected = {-55};

    ASSERT_EQ(result.size(), expected.size()) << "Matrix product has wrong size";

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(result[i], expected[i]) << "Matrix product has wrong value at index " << i;
    }
}

TEST(MatrixTests, columnByRowProductTest)
{
    const std::vector<double> A {-1, 0.5, 0.7, 1};
    const std::vector<double> B {3, 1, -2, 0.1};

    const std::vector<double> result = cascade::multiply(A, B, 4);

    const std::vector<double> expected = {-3, -1, 2, -0.1, 1.5, 0.5, -1, 0.05, 2.1, 0.7, -1.4, 0.07, 3, 1, -2, 0.1};

    ASSERT_EQ(result.size(), expected.size()) << "Matrix product has wrong size";

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(result[i], expected[i]) << "Matrix product has wrong value at index " << i;
    }
}
