#include "node_sqrt.h"

namespace cascade
{
void NodeSqrt::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() - (1.0 / (x->value() * x->value())) * derivative_);
}
}  // namespace cascade
