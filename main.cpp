#include "var.h"
#include <iostream>

int main()
{
    cascade::Var x = {1, 0.5};
    cascade::Var y = {2, 1};
    cascade::Var z = {3, 1.5};

    cascade::Var::setCovariance(x, y, -3);
    cascade::Var::setCovariance(x, z, 2.5);

    cascade::Var w = (x + y * z) * x;

    std::cout << w.index() << std::endl;
    std::cout << w.mean() << std::endl;
    std::cout << w.sigma() << std::endl;

    w.backprop();

    std::cout << x.derivative() << std::endl;
    std::cout << y.derivative() << std::endl;
    std::cout << z.derivative() << std::endl;

    return 0;
}
