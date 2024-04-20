#include "node_sinh.h"

#include <cmath>

namespace cascade
{
void NodeSinh::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + std::cosh(x->value()) * derivative_);
}
}  // namespace cascade
