#include "functions.h"
#include "tolerance.h"
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

TEST(DerivativeTests, derivativeOfAsin)
{
    const cascade::Var x = 0.55;

    const cascade::Var y = cascade::asin(x);
    y.backprop();

    // Wolfram Alpha: N[ReplaceAll[D[ArcSin[x], x], {x -> 0.55}], 16]
    EXPECT_DOUBLE_EQ(x.derivative(), 1.197368680178499) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfAcos)
{
    const cascade::Var x = -0.78;

    const cascade::Var y = cascade::acos(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), -1.598006930251483) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfAtan)
{
    const cascade::Var x = 187.2;

    const cascade::Var y = cascade::atan(x);
    y.backprop();

    EXPECT_NEAR(x.derivative(), 0.00002853487132485125, tolerance)
        << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfSinh)
{
    const cascade::Var x = 32.1;

    const cascade::Var y = cascade::sinh(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), std::cosh(x.value())) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfCosh)
{
    const cascade::Var x = -73.5;

    const cascade::Var y = cascade::cosh(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), std::sinh(x.value())) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfTanh)
{
    const cascade::Var x = 1.33;

    const cascade::Var y = cascade::tanh(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 1.0 / std::pow(std::cosh(x.value()), 2))
        << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfAsinh)
{
    const cascade::Var x = 3.78;

    const cascade::Var y = cascade::asinh(x);
    y.backprop();

    // Wolfram Alpha: N[ReplaceAll[D[ArcSinh[x], x], {x -> 3.78}], 16]
    EXPECT_DOUBLE_EQ(x.derivative(), 0.2557519663917190) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfAcosh)
{
    const cascade::Var x = 1.82;

    const cascade::Var y = cascade::acosh(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 0.6576101679733735) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfAtanh)
{
    const cascade::Var x = 23.45;

    const cascade::Var y = cascade::atanh(x);
    y.backprop();

    EXPECT_NEAR(x.derivative(), -0.001821817171537750, tolerance) << "Derivative has wrong value after backpropagation";
}
