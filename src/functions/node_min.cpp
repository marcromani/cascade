#include "node_min.h"

#include <cmath>

namespace cascade
{
void NodeMin::backprop_()
{
    std::shared_ptr<Node> x = children_.at(0);
    std::shared_ptr<Node> y = children_.at(1);

    const double x_ = x->value();
    const double y_ = y->value();

    // Compute a subgradient if x_ == y_
    const double dx = (x_ != y_) * (x_ < y_) + 0.5 * (x_ == y_);
    const double dy = (x_ != y_) * (x_ > y_) + 0.5 * (x_ == y_);

    x->setDerivative(x->derivative() + dx * derivative_);
    y->setDerivative(y->derivative() + dy * derivative_);
}
}  // namespace cascade
