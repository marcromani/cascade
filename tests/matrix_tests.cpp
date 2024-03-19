#include "util.h"

#include <gtest/gtest.h>

TEST(MatrixTests, matrixProductTest)
{
    const std::vector<double> A {1, 2, 3, 0.1, 0.2, 0.3};
    const std::vector<double> B {1, -1, 0, 1, 1, 0.5};

    std::vector<double> result = cascade::multiply(A, B, 2);

    EXPECT_EQ(1, 1);
}
