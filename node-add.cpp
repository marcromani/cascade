#include "node-add.h"

namespace cascade
{
void NodeAdd::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);
    std::shared_ptr<Node> y = children_.at(1);

    x->setDerivative(x->derivative() + derivative_);
    y->setDerivative(y->derivative() + derivative_);
}
}  // namespace cascade
