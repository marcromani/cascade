#include "cascade.h"

#include <iostream>

using namespace cascade;

// Rosenbrock function
Var f(Var x, Var y) { return pow(1.0 - x, 2.0) + 100.0 * pow(y - pow(x, 2.0), 2.0); }

int main()
{
    int numIterations   = 100000;
    double learningRate = 0.001;

    Var x = 0.5;
    Var y = -0.2;

    for (int i = 0; i < numIterations; ++i)
    {
        Var z = f(x, y);
        z.backprop();

        x = x.value() - learningRate * x.derivative();
        y = y.value() - learningRate * y.derivative();

        std::cout << z.value() << std::endl;
    }

    return 0;
}
