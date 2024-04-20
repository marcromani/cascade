#include "node_tanh.h"

#include <cmath>

namespace cascade
{
void NodeTanh::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    const double cosh = std::cosh(x->value());
    x->setDerivative(x->derivative() + (1.0 / (cosh * cosh)) * derivative_);
}
}  // namespace cascade
