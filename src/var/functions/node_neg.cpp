#include "node_neg.h"

#include <cmath>

namespace cascade
{
void NodeNeg::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() - derivative_);
}
}  // namespace cascade
