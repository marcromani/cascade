#include "node_abs.h"

#include <cmath>

namespace cascade
{
void NodeAbs::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    const int sign = (x->value() > 0.0) - (x->value() < 0.0);
    x->setDerivative(x->derivative() + sign * derivative_);
}
}  // namespace cascade
