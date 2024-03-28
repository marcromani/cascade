#include "node_atan.h"

#include <cmath>

namespace cascade
{
void NodeAtan::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);

    const double cos = std::cos(std::atan(x->value()));
    x->setDerivative(x->derivative() + cos * cos * derivative_);
}
}  // namespace cascade
