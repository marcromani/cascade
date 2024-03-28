#include "functions/functions.h"
#include "var.h"

#include <cmath>
#include <gtest/gtest.h>

TEST(DerivativeTests, variablesHaveZeroDerivativesAfterInitialization)
{
    const cascade::Var x;
    const cascade::Var y = 23.1;
    const cascade::Var z = {-2.2, 4.1};

    EXPECT_DOUBLE_EQ(x.derivative(), 0.0) << "Derivative is not zero after variable initialization";
    EXPECT_DOUBLE_EQ(y.derivative(), 0.0) << "Derivative is not zero after variable initialization";
    EXPECT_DOUBLE_EQ(z.derivative(), 0.0) << "Derivative is not zero after variable initialization";
}

TEST(DerivativeTests, derivativeOfSum)
{
    const cascade::Var x = 11.0;
    const cascade::Var y = -1.43;

    const cascade::Var z = x + y;
    z.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 1.0) << "Derivative has wrong value after backpropagation";
    EXPECT_DOUBLE_EQ(y.derivative(), 1.0) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfDifference)
{
    const cascade::Var x = -2.1;
    const cascade::Var y = 671.2;

    const cascade::Var z = x - y;
    z.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 1.0) << "Derivative has wrong value after backpropagation";
    EXPECT_DOUBLE_EQ(y.derivative(), -1.0) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfProduct)
{
    const cascade::Var x = 37.3;
    const cascade::Var y = 12.3;

    const cascade::Var z = x * y;
    z.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), y.value()) << "Derivative has wrong value after backpropagation";
    EXPECT_DOUBLE_EQ(y.derivative(), x.value()) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfDivision)
{
    const cascade::Var x = 13.2;
    const cascade::Var y = 0.05;

    const cascade::Var z = x / y;
    z.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 1.0 / y.value()) << "Derivative has wrong value after backpropagation";
    EXPECT_DOUBLE_EQ(y.derivative(), -x.value() / (y.value() * y.value()))
        << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfSin)
{
    const cascade::Var x = 41.4;

    const cascade::Var y = cascade::sin(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), std::cos(x.value())) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfCos)
{
    const cascade::Var x = 23.12;

    const cascade::Var y = cascade::cos(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), -std::sin(x.value())) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfTan)
{
    const cascade::Var x = 0.1;

    const cascade::Var y = cascade::tan(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 1.0 / std::pow(std::cos(x.value()), 2))
        << "Derivative has wrong value after backpropagation";
}
