#include "node-div.h"

namespace cascade
{
void NodeDiv::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);
    std::shared_ptr<Node> y = children_.at(1);

    x->setDerivative(x->derivative() + (1 / y->value()) * derivative_);
    y->setDerivative(y->derivative() - (x->value() / (y->value() * y->value())) * derivative_);
}
}  // namespace cascade
