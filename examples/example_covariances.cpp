#include "cascade.h"

#include <iostream>

using namespace cascade;

int main()
{
    // Create variables by providing their values and standard deviations (default to 0.0)
    Var x = {2.1, 1.5};
    Var y = {-3.5, 2.5};
    Var z = {5.7, 1.4};

    // Set the covariances between them (default to 0.0)
    Var::setCovariance(x, y, 0.5);
    Var::setCovariance(x, z, 1.8);
    Var::setCovariance(y, z, -1.0);

    // Compute a function of the variables
    Var f = (x + y) * cos(z) * x;

    bool changed = Var::setCovariance(f, x, 1.0);

    if (!changed)
    {
        std::cout << "Covariance involving a functional variable cannot be set" << std::endl;
    }

    // Computing covariances involving functional variables triggers backpropagation calls
    std::cout << Var::covariance(f, x) << std::endl;
    std::cout << Var::covariance(f, y) << std::endl;
    std::cout << Var::covariance(f, z) << std::endl;

    std::cout << Var::covariance(f, f) << std::endl;
    std::cout << f.sigma() * f.sigma() << std::endl;

    return 0;
}
