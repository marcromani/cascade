#include "var.h"

#include <iostream>

int main()
{
    cascade::Var x = {2, 1.5};
    cascade::Var y = {-3, 2.5};
    cascade::Var z = {5, 1.2};

    std::cout << cascade::Var::setCovariance(x, y, 0.5) << std::endl;
    std::cout << cascade::Var::setCovariance(x, z, 1.8) << std::endl;
    std::cout << cascade::Var::setCovariance(y, z, -1) << std::endl;

    cascade::Var w = (x + y) * z * x;

    std::cout << cascade::Var::setCovariance(w, x, 1.0) << std::endl;
    std::cout << cascade::Var::setCovariance(w, y, 1.0) << std::endl;
    std::cout << cascade::Var::setCovariance(w, z, 1.0) << std::endl;

    std::cout << cascade::Var::covariance(y, x) << std::endl;
    std::cout << cascade::Var::covariance(z, x) << std::endl;
    std::cout << cascade::Var::covariance(z, y) << std::endl;

    std::cout << w.sigma() * w.sigma() << std::endl;
    std::cout << cascade::Var::covariance(w, w) << std::endl;

    // std::vector<double> A {1, 2, 3, 0.1, 0.2, 0.3};
    // std::vector<double> B {1, -1, 0, 1, 1, 0.5};

    // std::vector<double> result = cascade::Var::matrixMultiply_(A, B, 2);

    return 0;
}
