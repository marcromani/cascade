#include "node_exp10.h"

#include <cmath>

namespace cascade
{
void NodeExp10::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + std::log(10.0) * value_ * derivative_);
}
}  // namespace cascade
