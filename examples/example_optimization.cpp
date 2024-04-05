#include "cascade.h"

#include <iostream>

using namespace cascade;

int main()
{
    Var x = 1.0;
    Var y = 0.5;

    Var z;

    for (int i = 0; i < 100; ++i)
    {
        std::cout << i << std::endl;
        z = x * y;
        x = x + 2.0;
    }

    z = x * y;

    z.backprop();

    return 0;
}
