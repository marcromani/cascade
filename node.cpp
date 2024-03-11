#include "node.h"

namespace cascade
{
    Node::Node() : Node(0.0) {}

    Node::Node(double value) : value_(value), derivative_(0.0) {}

    double Node::value() const { return value_; }

    double Node::derivative() const { return derivative_; }

    void Node::setDerivative(double derivative) { derivative_ = derivative; }

    void Node::backprop_() {}
}
