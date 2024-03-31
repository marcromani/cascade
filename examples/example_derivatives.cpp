#include "cascade.h"

#include <iostream>

using namespace cascade;

int main()
{
    Var x = 2.5;
    Var y = 1.2;
    Var z = 3.7;

    Var f = pow(x, 2.0) * sin(y) * exp(x / z);

    // Propagate the derivatives downstream
    f.backprop();

    // Recover the partial derivatives from the leaf nodes
    double fx = x.derivative();
    double fy = y.derivative();
    double fz = z.derivative();

    std::cout << "Value of f: " << f.value() << std::endl;
    std::cout << "Gradient of f: (" << fx << " " << fy << " " << fz << ")" << std::endl;

    return 0;
}
