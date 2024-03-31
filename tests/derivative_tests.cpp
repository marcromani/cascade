#include "functions.h"
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

TEST(DerivativeTests, backpropagationResetsDerivatives)
{
    const cascade::Var x = 2.0;
    const cascade::Var y = 1.0;

    const cascade::Var z = cascade::pow(y, 3.0);
    const cascade::Var w = 5.0 * x;

    const cascade::Var f = z + w;
    const cascade::Var g = z * w;

    f.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 5.0) << "Derivative has wrong value after backpropagation";
    EXPECT_DOUBLE_EQ(y.derivative(), 3.0 * y.value() * y.value()) << "Derivative has wrong value after backpropagation";

    g.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 5.0 * y.value() * y.value() * y.value())
        << "Derivative has wrong value after backpropagation";
    EXPECT_DOUBLE_EQ(y.derivative(), 15.0 * x.value() * y.value() * y.value())
        << "Derivative has wrong value after backpropagation";
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

TEST(DerivativeTests, derivativeOfNegation)
{
    const cascade::Var x = -876.3;

    const cascade::Var y = -x;
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), -1.0) << "Derivative has wrong value after backpropagation";
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

TEST(DerivativeTests, derivativeOfPow)
{
    const cascade::Var x = 1.3;
    const cascade::Var y = -3.7;

    const cascade::Var z = cascade::pow(x, y);
    z.backprop();

    // Wolfram Alpha: N[ReplaceAll[D[Pow[x, y], x], {x -> 1.3, y -> -3.7}], 16]
    EXPECT_DOUBLE_EQ(x.derivative(), -1.078122148810980) << "Derivative has wrong value after backpropagation";
    EXPECT_DOUBLE_EQ(y.derivative(), 0.09938349782502024) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfSqrt)
{
    const cascade::Var x = 144.0;

    const cascade::Var y = cascade::sqrt(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 0.04166666666666667) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfAbs)
{
    cascade::Var x = -45.75;

    cascade::Var y = cascade::abs(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), -1.0) << "Derivative has wrong value after backpropagation";

    x = 38.01;

    y = cascade::abs(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 1.0) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfExp)
{
    const cascade::Var x = -11.2;

    const cascade::Var y = cascade::exp(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), y.value()) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfExp2)
{
    const cascade::Var x = -11.2;

    const cascade::Var y = cascade::exp2(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 0.0002946385100017484) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfExp10)
{
    const cascade::Var x = 3.8;

    const cascade::Var y = cascade::exp10(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 14528.32975715202) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfLog)
{
    const cascade::Var x = 78.22;

    const cascade::Var y = cascade::log(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 1.0 / x.value()) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfLog2)
{
    const cascade::Var x = 34.99;

    const cascade::Var y = cascade::log2(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 0.04123163877933591) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfLog10)
{
    const cascade::Var x = 0.877;

    const cascade::Var y = cascade::log10(x);
    y.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 0.4952046543936737) << "Derivative has wrong value after backpropagation";
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

    EXPECT_DOUBLE_EQ(x.derivative(), 0.00002853487132485125) << "Derivative has wrong value after backpropagation";
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

    EXPECT_DOUBLE_EQ(x.derivative(), -0.001821817171537750) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfMin)
{
    cascade::Var x = 23.4;
    cascade::Var y = -12.1;

    cascade::Var z = cascade::min(x, y);
    z.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 0.0) << "Derivative has wrong value after backpropagation";
    EXPECT_DOUBLE_EQ(y.derivative(), 1.0) << "Derivative has wrong value after backpropagation";

    x = 34.27;
    y = 34.27;

    z = cascade::min(x, y);
    z.backprop();

    // Check subgradient
    EXPECT_DOUBLE_EQ(x.derivative(), 0.5) << "Derivative has wrong value after backpropagation";
    EXPECT_DOUBLE_EQ(y.derivative(), 0.5) << "Derivative has wrong value after backpropagation";
}

TEST(DerivativeTests, derivativeOfMax)
{
    cascade::Var x = 11.1;
    cascade::Var y = 0.55;

    cascade::Var z = cascade::max(x, y);
    z.backprop();

    EXPECT_DOUBLE_EQ(x.derivative(), 1.0) << "Derivative has wrong value after backpropagation";
    EXPECT_DOUBLE_EQ(y.derivative(), 0.0) << "Derivative has wrong value after backpropagation";

    x = 42.8;
    y = 42.8;

    z = cascade::max(x, y);
    z.backprop();

    // Check subgradient
    EXPECT_DOUBLE_EQ(x.derivative(), 0.5) << "Derivative has wrong value after backpropagation";
    EXPECT_DOUBLE_EQ(y.derivative(), 0.5) << "Derivative has wrong value after backpropagation";
}
