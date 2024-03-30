#include "node_atan.h"

#include <cmath>

namespace cascade
{
void NodeAtan::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + (1.0 / (1 + x->value() * x->value())) * derivative_);
}
}  // namespace cascade
