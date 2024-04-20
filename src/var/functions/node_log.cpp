#include "node_log.h"

namespace cascade
{
void NodeLog::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    x->setDerivative(x->derivative() + (1.0 / x->value()) * derivative_);
}
}  // namespace cascade
