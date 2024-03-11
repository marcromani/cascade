#include "var.h"
#include <iostream>

int main()
{
    cascade::Var x = {1, 0.5};
    cascade::Var y = {2, 1};
    cascade::Var z = {3, 1.5};

    cascade::Var::setCovariance(x, y, -3);
    cascade::Var::setCovariance(x, z, 2.5);

    cascade::Var v = (x + y) * z;
    cascade::Var w = v * v * x;

    std::cout << w.id() << std::endl;
    std::cout << w.value() << std::endl;
    std::cout << w.sigma() << std::endl;

    w.backprop();

    std::cout << x.derivative() << std::endl;
    std::cout << y.derivative() << std::endl;
    std::cout << z.derivative() << std::endl;
    std::cout << v.derivative() << std::endl;

    cascade::Var t = w;
    x = w;

    std::cout << w.id() << std::endl;
    std::cout << t.id() << std::endl;
    std::cout << x.id() << std::endl;

    x = x + w;
    std::cout << x.id() << std::endl;

    cascade::Var a = x + y;
    cascade::Var b = w + a;

    std::cout << cascade::Var::covariance(z, z) << std::endl;

    return 0;
}
