#include "node_sin.h"

#include <cmath>

namespace cascade
{
void NodeSin::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + std::cos(x->value()) * derivative_);
}
}  // namespace cascade
