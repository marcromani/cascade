#include "node_acos.h"

#include <cmath>

namespace cascade
{
void NodeAcos::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() - (1.0 / std::sqrt(1.0 - x->value() * x->value())) * derivative_);
}
}  // namespace cascade
