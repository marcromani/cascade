#include "node_asin.h"

#include <cmath>

namespace cascade
{
void NodeAsin::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + (1.0 / std::sin(std::asin(x->value()))) * derivative_);
}
}  // namespace cascade
