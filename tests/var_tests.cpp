#include "functions.h"
#include "var.h"

#include <gtest/gtest.h>

TEST(VarTests, variablesAreShallowCopiedInAssignment)
{
    const cascade::Var x = 23.3;
    const cascade::Var y = 71.6;

    const cascade::Var z = cascade::exp(x * y) * cascade::cos(10.5 * x);
    z.backprop();

    const double zx = x.derivative();
    const double zy = y.derivative();

    const cascade::Var w = z;
    w.backprop();

    const double wx = x.derivative();
    const double wy = y.derivative();

    EXPECT_EQ(w.id(), z.id()) << "Variables do not have the same id";

    EXPECT_DOUBLE_EQ(wx, zx) << "Derivatives are not equal";
    EXPECT_DOUBLE_EQ(wy, zy) << "Derivatives are not equal";
}

TEST(VarTests, variablesAreShallowCopiedInConstructor)
{
    const cascade::Var x = 1.7;
    const cascade::Var y = -4.9;

    const cascade::Var z = cascade::tanh(x * cascade::abs(y)) + cascade::exp(y / x);
    z.backprop();

    const double zx = x.derivative();
    const double zy = y.derivative();

    const cascade::Var w(z);
    w.backprop();

    const double wx = x.derivative();
    const double wy = y.derivative();

    EXPECT_EQ(w.id(), z.id()) << "Variables do not have the same id";

    EXPECT_DOUBLE_EQ(wx, zx) << "Derivatives are not equal";
    EXPECT_DOUBLE_EQ(wy, zy) << "Derivatives are not equal";
}
