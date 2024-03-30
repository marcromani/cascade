#include "node_log2.h"

#include <cmath>

namespace cascade
{
void NodeLog2::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + (1.0 / (std::log(2.0) * x->value())) * derivative_);
}
}  // namespace cascade
