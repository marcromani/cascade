<div align="center">
<img src="https://raw.githubusercontent.com/marcromani/cascade/main/media/logo.png" alt="Library logo" width="600"/>
</div>

#

**Cascade** is a C++ library designed for error propagation using automatic differentiation. It allows users to compute gradients of an expression and then use them to propagate uncertainties in the input to the output. The library simplifies the process of error analysis in scientific computing and engineering applications.

## How to build

Create a `build` folder in the root directory and `cd` it. Build the library and tests executable with:

```
cmake ..
cmake --build .
```

The library is built in `build/src` and the tests executable in `build/tests`.

## How to install

To further install the library do:

```
cmake --build . --target install
```

If you want to install the library in a custom directory set the install path first:

```
cmake -DCMAKE_INSTALL_PREFIX=/install/path ..
cmake --build . --target install
```

To use Cascade in your project simply include the Cascade header files and link against the library.

## Examples

### Source

```c++
#include "math.h"
#include "var.h"

#include <iostream>

int main()
{
    cascade::Var x = {2, 1.5};
    cascade::Var y = {-3, 2.5};
    cascade::Var z = {5, 1.2};

    cascade::Var::setCovariance(x, y, 0.5);
    cascade::Var::setCovariance(x, z, 1.8);

    if (!cascade::Var::setCovariance(y, z, -1))
    {
        std::cout << "This should not be printed" << std::endl;
    }

    cascade::Var w = (x + y) * cascade::cos(z) * x;

    if (!cascade::Var::setCovariance(w, x, 1.0))
    {
        std::cout << "Covariance involving a functional variable cannot be set" << std::endl;
    }

    std::cout << cascade::Var::covariance(w, x) << std::endl;
    std::cout << cascade::Var::covariance(w, y) << std::endl;
    std::cout << cascade::Var::covariance(w, z) << std::endl;

    std::cout << cascade::Var::covariance(w, w) << std::endl;
    std::cout << w.sigma() * w.sigma() << std::endl;

    return 0;
}
```

### Output

```
Covariance involving a functional variable cannot be set
-2.53023
5.60546
-2.81843
7.86771
7.86771
```

## License

Cascade is licensed under the [MIT License](LICENSE).

## Background

Cascade uses reverse mode automatic differentiation, most commonly known as backpropagation, to compute exact derivatives of arbitrary (piecewise) differentiable functions. The key concept behind this technique is compositionality. Each function defines a (acyclic) computational graph where its nodes store the intermediate values of the operations that make up the result, and each edge stores the derivative of a parent node with respect to a child node. One can then efficiently apply the chain rule on the function by sorting the graph nodes in a topological order and allow the derivatives to flow backwards, from the output to the inputs. For example,

<div align="center">
<img src="https://raw.githubusercontent.com/marcromani/cascade/doc/media/graph.png" alt="Function graph"/>
</div>

Here each node is given a name and is already indexed following a topological order. The partial derivative of the function with respect to $x$ corresponds to adding all the paths from the output node to the $x$ node, where the value of each path is the product of the values of its edges:

<div align="center">
<img src="https://raw.githubusercontent.com/marcromani/cascade/doc/media/derivative_graph.png" alt="Function derivative graph"/>
</div>

```math
\begin{align*}
\frac{\partial\omega_9}{\partial x} = & \, \frac{\partial\omega_9}{\partial\omega_7}\frac{\partial\omega_7}{\partial x} + \frac{\partial\omega_9}{\partial\omega_8}\frac{\partial\omega_8}{\partial x} \\
= & \, \frac{\partial\omega_9}{\partial\omega_7} \left(\frac{\partial\omega_7}{\partial\omega_4}\frac{\partial\omega_4}{\partial x} + \frac{\partial\omega_7}{\partial\omega_5}\frac{\partial\omega_5}{\partial x}\right) + \frac{\partial\omega_9}{\partial\omega_8}\left(\frac{\partial\omega_8}{\partial\omega_6}\frac{\partial\omega_6}{\partial x}\right) \\
= & \, \frac{\partial\omega_9}{\partial\omega_7} \left(\frac{\partial\omega_7}{\partial\omega_4}\left(\frac{\partial\omega_4}{\partial\omega_2}\frac{\partial\omega_2}{\partial x}\right) + \frac{\partial\omega_7}{\partial\omega_5}\left(\frac{\partial\omega_5}{\partial\omega_1}\frac{\partial\omega_1}{\partial x}\right)\right) + \frac{\partial\omega_9}{\partial\omega_8}\left(\frac{\partial\omega_8}{\partial\omega_6}\left(\frac{\partial\omega_6}{\partial\omega_1}\frac{\partial\omega_1}{\partial x} + \frac{\partial\omega_6}{\partial\omega_3}\frac{\partial\omega_3}{\partial x}\right)\right) \\
= & \, \frac{\partial\omega_9}{\partial\omega_7} \left(\frac{\partial\omega_7}{\partial\omega_4}\left(\frac{\partial\omega_4}{\partial\omega_2}\cdot0\right) + \frac{\partial\omega_7}{\partial\omega_5}\left(\frac{\partial\omega_5}{\partial\omega_1}\cdot1\right)\right) + \frac{\partial\omega_9}{\partial\omega_8}\left(\frac{\partial\omega_8}{\partial\omega_6}\left(\frac{\partial\omega_6}{\partial\omega_1}\cdot1 + \frac{\partial\omega_6}{\partial\omega_3}\cdot0\right)\right) \\
= & \, \frac{\partial\omega_9}{\partial\omega_7}\frac{\partial\omega_7}{\partial\omega_5}\frac{\partial\omega_5}{\partial\omega_1} + \frac{\partial\omega_9}{\partial\omega_8}\frac{\partial\omega_8}{\partial\omega_6}\frac{\partial\omega_6}{\partial\omega_1}\,.
\end{align*}
```

Suppose you have a set of $n$ distinct variables that you observe over time, each of which comes with some error. If you can further estimate the size of these errors and the size of the correlations between the variables, you can model the source of your observations as a random vector $\boldsymbol{X} = (X_1, \ldots, X_n)$ with expected value

```math
\text{E}\left[\boldsymbol{X}\right] = (\text{E}\left[X_1\right], \ldots, \text{E}\left[X_n\right]) = (\mu_1, \ldots, \mu_n) = \boldsymbol{\mu}
```

and covariance matrix

```math
\text{Cov}\left[\boldsymbol{X}\right] = \left(\begin{matrix}\text{Cov}\left[X_1, X_1\right] & \text{Cov}\left[X_1, X_2\right] & \cdots & \text{Cov}\left[X_1, X_n\right] \cr \text{Cov}\left[X_2, X_1\right] & \text{Cov}\left[X_2, X_2\right] & \cdots & \text{Cov}\left[X_2, X_n\right] \cr \vdots & \vdots & \ddots & \vdots \cr \text{Cov}\left[X_n, X_1\right] & \text{Cov}\left[X_n, X_2\right] & & \text{Cov}\left[X_n, X_n\right]\end{matrix}\right) = \left(\begin{matrix}{\sigma_1}^2 & \sigma_{12} & \cdots & \sigma_{1n} \cr \sigma_{21} & {\sigma_2}^2 & \cdots & \sigma_{2n} \cr \vdots & \vdots & \ddots & \vdots \cr \sigma_{n1} & \sigma_{n2} & \cdots & {\sigma_{n}}^2\end{matrix}\right)\,.
```

Whenever you compute a value $Y$ that is a function of these variables, say $Y = f(\boldsymbol{X})$, their uncertainties will be propagated to the result, so that it will also be a random variable. An estimation of the error associated with this new variable can be computed by linearizing the function around the expected value of the input, $\boldsymbol{\mu}$:

```math
\begin{align*}
\text{Var}\left[Y\right] \approx & \, \text{Var}\left[f(\boldsymbol{\mu}) + {\nabla{f}(\boldsymbol{\mu})}^\top(\boldsymbol{X} - \boldsymbol{\mu})\right] \\
= & \, \text{Var}\left[{\nabla{f}(\boldsymbol{\mu})}^\top(\boldsymbol{X} - \boldsymbol{\mu})\right] \\
= & \, \text{E}\left[{\nabla{f}(\boldsymbol{\mu})}^\top(\boldsymbol{X} - \boldsymbol{\mu}){(\boldsymbol{X} - \boldsymbol{\mu})}^\top\nabla{f}(\boldsymbol{\mu})\right] \\
= & \, {\nabla{f}(\boldsymbol{\mu})}^\top\text{E}\left[(\boldsymbol{X} - \boldsymbol{\mu}){(\boldsymbol{X} - \boldsymbol{\mu})}^\top\right]\nabla{f}(\boldsymbol{\mu}) \\
= & \, {\nabla{f}(\boldsymbol{\mu})}^\top\text{Cov}\left[\boldsymbol{X}\right]\nabla{f}(\boldsymbol{\mu})\,.
\end{align*}
```

For uncorrelated variables one recovers the well-known formula for error propagation:

```math
\text{Var}\left[Y\right] \approx \sum_{i=1}^n\left({\frac{\partial f}{\partial x_i}(\boldsymbol{\mu})}\right)^2{\sigma_i}^2\,.
```

The same idea can also be used to compute the covariance between two variables that are both functions of $\boldsymbol{X}$. For instance, if $Z = g(\boldsymbol{X})$:

```math
\begin{align*}
\text{Cov}\left[Y, Z\right] = & \, \text{E}\left[(Y - f(\boldsymbol{\mu}))(Z - g(\boldsymbol{\mu}))\right] \\
\approx & \, \text{E}\left[{\nabla{f}(\boldsymbol{\mu})}^\top(\boldsymbol{X} - \boldsymbol{\mu}){\nabla{g}(\boldsymbol{\mu})}^\top(\boldsymbol{X} - \boldsymbol{\mu})\right] \\
= & \, \text{E}\left[{\nabla{f}(\boldsymbol{\mu})}^\top(\boldsymbol{X} - \boldsymbol{\mu}){(\boldsymbol{X} - \boldsymbol{\mu})}^\top\nabla{g}(\boldsymbol{\mu})\right] \\
= & \, {\nabla{f}(\boldsymbol{\mu})}^\top\text{E}\left[(\boldsymbol{X} - \boldsymbol{\mu}){(\boldsymbol{X} - \boldsymbol{\mu})}^\top\right]\nabla{g}(\boldsymbol{\mu}) \\
= & \, {\nabla{f}(\boldsymbol{\mu})}^\top\text{Cov}\left[\boldsymbol{X}\right]\nabla{g}(\boldsymbol{\mu})\,.
\end{align*}
```
