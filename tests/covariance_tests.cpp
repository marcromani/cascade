#include "functions.h"
#include "tolerance.h"
#include "util.h"
#include "var.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

TEST(CovarianceTests, variablesHaveZeroSigmaIfNotProvidedAtInitialization)
{
    const cascade::Var x;
    const cascade::Var y = 83.6;

    EXPECT_DOUBLE_EQ(x.sigma(), 0.0) << "Sigma is not zero after variable initialization";
    EXPECT_DOUBLE_EQ(y.sigma(), 0.0) << "Sigma is not zero after variable initialization";
}

TEST(CovarianceTests, sigmaAndVarianceConsistencyForLeafNodeTest)
{
    cascade::Var x = {10.0, 1.5};

    EXPECT_DOUBLE_EQ(x.sigma() * x.sigma(), cascade::Var::covariance(x, x))
        << "The standard deviation is not the square root of the variance";

    EXPECT_EQ(x.setSigma(22.0), true) << "Sigma setter returns `false` on a functional node";

    EXPECT_DOUBLE_EQ(x.sigma() * x.sigma(), cascade::Var::covariance(x, x))
        << "The standard deviation is not the square root of the variance";
}

TEST(CovarianceTests, sigmaAndVarianceConsistencyForFunctionNodeTest)
{
    cascade::Var x = {2.0, 1.5};
    cascade::Var y = {-3.0, 2.5};
    cascade::Var z = {5.0, 1.2};

    cascade::Var::setCovariance(x, y, 0.5);
    cascade::Var::setCovariance(x, z, 1.8);
    cascade::Var::setCovariance(y, z, -1.0);

    const cascade::Var w = (x + y) * z * x;

    EXPECT_DOUBLE_EQ(w.sigma() * w.sigma(), cascade::Var::covariance(w, w))
        << "The standard deviation is not the square root of the variance";
}

TEST(CovarianceTests, sigmaIsFixedForFunctionNodeTest)
{
    const cascade::Var x = {1.0, 3.0};
    const cascade::Var y = {2.0, 0.2};

    cascade::Var z = x * y;

    EXPECT_EQ(z.setSigma(1.5), false) << "Sigma setter returns `true` on a functional node";

    EXPECT_DOUBLE_EQ(z.sigma() * z.sigma(), 36.04) << "Sigma is manually changed on a functional node";
}

TEST(CovarianceTests, varianceOfLinearFunctionNodeTest)
{
    cascade::Var x = 2.0;
    cascade::Var y = -1.0;
    cascade::Var z = 0.1;

    x.setSigma(1.0);
    y.setSigma(3.0);
    z.setSigma(2.0);

    cascade::Var::setCovariance(x, y, 0.1);
    cascade::Var::setCovariance(x, z, 2.0);
    cascade::Var::setCovariance(y, z, 1.0);

    const cascade::Var w = -2.0 * x + 3.0 * y + z - 12.0;

    const double expectedVariance = 85.8;

    EXPECT_DOUBLE_EQ(w.sigma() * w.sigma(), expectedVariance) << "Wrong value after variance propagation";
}

TEST(CovarianceTests, varianceOfNonlinearFunctionNodeTest)
{
    cascade::Var x = {2.0, 1.0};
    cascade::Var y = {-1.0, 2.0};
    cascade::Var z = {-2.0, 3.0};

    cascade::Var::setCovariance(x, y, 0.5);
    cascade::Var::setCovariance(x, z, -1.0);
    cascade::Var::setCovariance(y, z, 0.2);

    const cascade::Var w = (x * x * y + z) * y;

    const double expectedVariance = 397.0;

    EXPECT_DOUBLE_EQ(w.sigma() * w.sigma(), expectedVariance) << "Wrong value after variance propagation";
}

TEST(CovarianceTests, covarianceOfNonlinearFunctionNodesTest)
{
    cascade::Var x = {-1.0, 1.0};
    cascade::Var y = {3.0, 0.5};
    cascade::Var z = {0.5, 2.0};

    cascade::Var::setCovariance(x, y, -1.0);
    cascade::Var::setCovariance(x, z, 0.5);
    cascade::Var::setCovariance(y, z, 1.0);

    const cascade::Var f = (x + z) * cascade::sin(x * y + z) / y;
    const cascade::Var g = (x + z) * x * y;

    const double covariance = cascade::Var::covariance(f, g);

    const double x_ = x.value();
    const double y_ = y.value();
    const double z_ = z.value();

    const double fx = std::sin(x_ * y_ + z_) / y_ + (x_ + z_) * std::cos(x_ * y_ + z_);
    const double fy = (x_ + z_) * (std::cos(x_ * y_ + z_) * x_ * y_ - std::sin(x_ * y_ + z_)) / (y_ * y_);
    const double fz = (std::sin(x_ * y_ + z_) + (x_ + z_) * std::cos(x_ * y_ + z_)) / y_;

    const std::vector<double> fGrad = {fx, fy, fz};

    const double gx = (x_ * y_) + (x_ + z_) * y_;
    const double gy = (x_ + z_) * x_;
    const double gz = x_ * y_;

    const std::vector<double> gGrad = {gx, gy, gz};

    const std::vector<double> M = {1.0, -1.0, 0.5, -1.0, 0.25, 1.0, 0.5, 1.0, 4.0};

    std::vector<double> expectedCovariance = cascade::multiply(M, fGrad, 3);
    expectedCovariance                     = cascade::multiply(gGrad, expectedCovariance, 1);

    EXPECT_NEAR(covariance, expectedCovariance[0], tolerance) << "Wrong value after variance propagation";
}
