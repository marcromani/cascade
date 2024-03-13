#include "node.h"

namespace cascade
{
int Node::counter_ = 0;

Node::Node() : Node(0.0) {}

Node::Node(double value) : id_(counter_), value_(value), derivative_(0.0) { ++counter_; }

int Node::id() const { return id_; }

double Node::value() const { return value_; }

double Node::derivative() const { return derivative_; }

void Node::setDerivative(double derivative) { derivative_ = derivative; }
}  // namespace cascade
