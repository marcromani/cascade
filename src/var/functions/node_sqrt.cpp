#include "node_sqrt.h"

namespace cascade
{
void NodeSqrt::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + (0.5 / value_) * derivative_);
}
}  // namespace cascade
