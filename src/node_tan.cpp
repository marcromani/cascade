#include "functions/node_tan.h"

#include <cmath>

namespace cascade
{
void NodeTan::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    const double cos = std::cos(x->value());
    x->setDerivative(x->derivative() + (1.0 / (cos * cos)) * derivative_);
}
}  // namespace cascade
