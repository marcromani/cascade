#include "node_exp2.h"

#include <cmath>

namespace cascade
{
void NodeExp2::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + std::log(2.0) * value_ * derivative_);
}
}  // namespace cascade
