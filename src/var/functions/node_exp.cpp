#include "node_exp.h"

namespace cascade
{
void NodeExp::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + value_ * derivative_);
}
}  // namespace cascade
