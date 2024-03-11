#include "var.h"

#include <iostream>

int main()
{
    cascade::Var x = {2, 1.5};
    cascade::Var y = {-3, 2.5};
    cascade::Var z = {5, 1.2};

    cascade::Var::setCovariance(x, y, 0.5);
    cascade::Var::setCovariance(x, z, 1.8);
    cascade::Var::setCovariance(y, z, -1);

    cascade::Var w = (x + y) * z * x;

    cascade::Var::covariance2(w, w);

    return 0;
}
