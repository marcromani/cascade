#include "var.h"
#include <iostream>

int main()
{
    cascade::Var x = {3.5, 1.2};
    cascade::Var y = 10;

    std::cout << x << std::endl;
    std::cout << y << std::endl;

    std::cout << x.index() << std::endl;
    std::cout << y.index() << std::endl;

    cascade::Var::setCovariance(y, x, -2);
    std::cout << cascade::Var::covariance(y, x) << std::endl;

    cascade::Var z = {4, 0.25};
    std::cout << cascade::Var::covariance(z, x) << std::endl;

    cascade::Var w = x;
    std::cout << w.index() << std::endl;
    std::cout << cascade::Var::covariance(w, y) << std::endl;

    w = x;
    std::cout << w.index() << std::endl;
    std::cout << cascade::Var::covariance(w, y) << std::endl;

    cascade::Var a = 1;
    cascade::Var b = 2;

    cascade::Var::setCovariance(a, b, 10);
    std::cout << cascade::Var::covariance(a, b) << std::endl;

    b = x;
    std::cout << cascade::Var::covariance(a, b) << std::endl;

    y = x;
    std::cout << y << std::endl;
    cascade::Var r = y + x;
    std::cout << r << std::endl;
    r = x + y;
    std::cout << r << std::endl;

    std::cout << cascade::Var::covariance(r, x) << std::endl;

    return 0;
}
