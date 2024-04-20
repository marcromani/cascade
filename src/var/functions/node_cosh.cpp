#include "node_cosh.h"

#include <cmath>

namespace cascade
{
void NodeCosh::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + std::sinh(x->value()) * derivative_);
}
}  // namespace cascade
