#include "node.h"

namespace cascade
{
    int Node::counter_ = 0;

    Node::Node() : Node(0.0) {}

    Node::Node(double value) : Node(value, 0.0) {}

    Node::Node(double value, double sigma) : id_(counter_), value_(value), sigma_(sigma), derivative_(0.0)
    {
        covariance_.emplace(id_, sigma * sigma);

        ++counter_;
    }

    int Node::id() const { return id_; }

    double Node::value() const { return value_; }

    double Node::sigma() const { return sigma_; }

    double Node::derivative() const { return derivative_; }

    void Node::setDerivative(double derivative) { derivative_ = derivative; }

    double Node::covariance(const std::shared_ptr<Node> x, const std::shared_ptr<Node> y)
    {
        const Node *parent;
        const Node *child;

        if (x->id_ < y->id_)
        {
            parent = x.get();
            child = y.get();
        }
        else
        {
            parent = y.get();
            child = x.get();
        }

        auto search = parent->covariance_.find(child->id_);

        if (search != parent->covariance_.end())
        {
            return search->second;
        }
        else
        {
            return 0.0;
        }
    }

    void Node::setCovariance(std::shared_ptr<Node> x, std::shared_ptr<Node> y, double value)
    {
        if (x->id_ < y->id_)
        {
            x->covariance_.emplace(y->id_, value);
        }
        else
        {
            y->covariance_.emplace(x->id_, value);
        }
    }

    void Node::backprop_() {}
}
