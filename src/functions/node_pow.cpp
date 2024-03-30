#include "node_pow.h"

#include <cmath>

namespace cascade
{
void NodePow::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);
    std::shared_ptr<Node> y = children_.at(1);

    x->setDerivative(x->derivative() + y->value() * std::pow(x->value(), y->value() - 1.0) * derivative_);
    y->setDerivative(y->derivative() + std::log(x->value()) * value_ * derivative_);
}
}  // namespace cascade
