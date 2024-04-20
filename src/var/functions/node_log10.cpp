#include "node_log10.h"

#include <cmath>

namespace cascade
{
void NodeLog10::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + (1.0 / (std::log(10.0) * x->value())) * derivative_);
}
}  // namespace cascade
