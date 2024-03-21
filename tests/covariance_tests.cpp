#include "var.h"

#include <gtest/gtest.h>

TEST(CovarianceTests, sigmaAndVarianceConsistencyForLeafNodeTest)
{
    cascade::Var x = {10, 1.5};

    EXPECT_DOUBLE_EQ(x.sigma() * x.sigma(), cascade::Var::covariance(x, x))
        << "The standard deviation is not the square root of the variance";

    x.setSigma(22);

    EXPECT_DOUBLE_EQ(x.sigma() * x.sigma(), cascade::Var::covariance(x, x))
        << "The standard deviation is not the square root of the variance";
}

TEST(CovarianceTests, sigmaAndVarianceConsistencyForFunctionNodeTest)
{
    cascade::Var x = {2, 1.5};
    cascade::Var y = {-3, 2.5};
    cascade::Var z = {5, 1.2};

    cascade::Var::setCovariance(x, y, 0.5);
    cascade::Var::setCovariance(x, z, 1.8);
    cascade::Var::setCovariance(y, z, -1);

    const cascade::Var w = (x + y) * z * x;

    EXPECT_DOUBLE_EQ(w.sigma() * w.sigma(), cascade::Var::covariance(w, w))
        << "The standard deviation is not the square root of the variance";
}

TEST(CovarianceTests, sigmaIsFixedForFunctionNodeTest)
{
    const cascade::Var x = {1, 3};
    const cascade::Var y = {2, 0.2};

    cascade::Var z = x * y;

    EXPECT_EQ(z.setSigma(1.5), false) << "Sigma setter returns `true` on a functional node";
    EXPECT_DOUBLE_EQ(z.sigma() * z.sigma(), 36.04) << "Sigma is manually changed on a functional node";
}

TEST(CovarianceTests, varianceOfLinearFunctionNodeTest)
{
    cascade::Var x = 2;
    cascade::Var y = -1;
    cascade::Var z = 0.1;

    x.setSigma(1);
    y.setSigma(3);
    z.setSigma(2);

    cascade::Var::setCovariance(x, y, 0.1);
    cascade::Var::setCovariance(x, z, 2);
    cascade::Var::setCovariance(y, z, 1);

    const cascade::Var w = -2 * x + 3 * y + z;

    const double expectedVariance = 85.8;

    EXPECT_DOUBLE_EQ(w.sigma() * w.sigma(), expectedVariance) << "Wrong value after variance propagation";
}

TEST(CovarianceTests, varianceOfNonlinearFunctionNodeTest)
{
    cascade::Var x = {2, 1};
    cascade::Var y = {-1, 2};
    cascade::Var z = {-2, 3};

    cascade::Var::setCovariance(x, y, 0.5);
    cascade::Var::setCovariance(x, z, -1);
    cascade::Var::setCovariance(y, z, 0.2);

    const cascade::Var w = (x * x * y + z) * y;

    const double expectedVariance = 397;

    EXPECT_DOUBLE_EQ(w.sigma() * w.sigma(), expectedVariance) << "Wrong value after variance propagation";
}

TEST(CovarianceTests, covarianceOfNonlinearFunctionNodesTest) {}
