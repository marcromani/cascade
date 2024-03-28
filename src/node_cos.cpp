#include "functions/node_cos.h"

#include <cmath>

namespace cascade
{
void NodeCos::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() - std::sin(x->value()) * derivative_);
}
}  // namespace cascade
