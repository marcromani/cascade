#include "node_asinh.h"

#include <cmath>

namespace cascade
{
void NodeAsinh::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + (1.0 / std::sqrt(x->value() * x->value() + 1.0)) * derivative_);
}
}  // namespace cascade
