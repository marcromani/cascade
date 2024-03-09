#include "unumber.h"
#include <iostream>

int main()
{
    UDouble x = {3.5, 1.2};
    UInt y = 10;

    std::cout << x << std::endl;
    std::cout << y << std::endl;

    std::cout << x.index() << std::endl;
    std::cout << y.index() << std::endl;

    setCovariance(y, x, -2);
    std::cout << covariance(y, x) << std::endl;

    UFloat z = {4, 0.25};
    std::cout << covariance(z, x) << std::endl;

    UDouble w = x;
    std::cout << w.index() << std::endl;
    std::cout << covariance(w, y) << std::endl;

    w = x;
    std::cout << w.index() << std::endl;
    std::cout << covariance(w, y) << std::endl;

    UDouble a = 1;
    UDouble b = 2;

    setCovariance(a, b, 10);
    std::cout << covariance(a, b) << std::endl;

    b = x;
    std::cout << covariance(a, b) << std::endl;

    y = x;
    std::cout << y << std::endl;
    UInt r = y + x;
    std::cout << r << std::endl;
    r = x + y;
    std::cout << r << std::endl;

    return 0;
}
