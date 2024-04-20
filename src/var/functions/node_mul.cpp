#include "node_mul.h"

namespace cascade
{
void NodeMul::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);
    std::shared_ptr<Node> y = children_.at(1);

    x->setDerivative(x->derivative() + y->value() * derivative_);
    y->setDerivative(y->derivative() + x->value() * derivative_);
}
}  // namespace cascade
