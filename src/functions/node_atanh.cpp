#include "node_atanh.h"

#include <cmath>

namespace cascade
{
void NodeAtanh::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + (1.0 / (1.0 - x->value() * x->value())) * derivative_);
}
}  // namespace cascade
