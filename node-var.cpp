#include "node-var.h"

#include <cmath>
#include <memory>

namespace cascade
{
NodeVar::NodeVar() : NodeVar(0.0) {}

NodeVar::NodeVar(double value) : NodeVar(value, 0.0) {}

NodeVar::NodeVar(double value, double sigma) : Node(value)
{
    covariance_.emplace(id_, sigma * sigma);

    ++counter_;
}

double NodeVar::sigma() const
{
    return std::sqrt(covariance(std::make_shared<NodeVar>(*this), std::make_shared<NodeVar>(*this)));
}

double NodeVar::covariance(const std::shared_ptr<NodeVar>& x, const std::shared_ptr<NodeVar>& y)
{
    const NodeVar* parent;
    const NodeVar* child;

    if (x->id_ < y->id_)
    {
        parent = x.get();
        child  = y.get();
    }
    else
    {
        parent = y.get();
        child  = x.get();
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

void NodeVar::setCovariance(std::shared_ptr<NodeVar>& x, std::shared_ptr<NodeVar>& y, double value)
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

void NodeVar::backprop_() {}
}  // namespace cascade
