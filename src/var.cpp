#include "var.h"

#include "functions/node_add.h"
#include "functions/node_div.h"
#include "functions/node_mul.h"
#include "functions/node_neg.h"
#include "functions/node_sub.h"
#include "node.h"
#include "node_var.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <ostream>
#include <set>
#include <stack>
#include <vector>

namespace cascade
{
Var::Var() : Var(0.0) {}

Var::Var(double value) : Var(value, 0.0) {}

Var::Var(double value, double sigma) : node_(std::make_shared<NodeVar>(value, sigma)) {}

int Var::id() const { return node_->id_; }

double Var::value() const { return node_->value_; }

double Var::sigma() const
{
    const std::shared_ptr<NodeVar> node = std::dynamic_pointer_cast<NodeVar>(node_);

    if (node)
    {
        return node->sigma();
    }
    else
    {
        return std::sqrt(covariance_(*this, *this));
    }
}

double Var::derivative() const { return node_->derivative_; }

bool Var::setSigma(double value) { return setCovariance(*this, *this, value * value); }

double Var::covariance(const Var &x, const Var &y)
{
    const std::shared_ptr<NodeVar> xNode = std::dynamic_pointer_cast<NodeVar>(x.node_);
    const std::shared_ptr<NodeVar> yNode = std::dynamic_pointer_cast<NodeVar>(y.node_);

    if (xNode && yNode)
    {
        return NodeVar::covariance(xNode, yNode);
    }
    else
    {
        return covariance_(x, y);
    }
}

bool Var::setCovariance(Var &x, Var &y, double value)
{
    std::shared_ptr<NodeVar> xNode = std::dynamic_pointer_cast<NodeVar>(x.node_);
    std::shared_ptr<NodeVar> yNode = std::dynamic_pointer_cast<NodeVar>(y.node_);

    if (xNode && yNode)
    {
        NodeVar::setCovariance(xNode, yNode, value);
        return true;
    }
    else
    {
        return false;
    }
}

void Var::backprop() const
{
    const std::vector<Var> nodes = sortedNodes_();

    for (const Var &node: nodes)
    {
        node.node_->derivative_ = 0.0;
    }

    node_->derivative_ = 1.0;

    for (const Var &node: nodes)
    {
        node.node_->backprop_();
    }
}

Var operator+(Var x, Var y)
{
    Var result = x.value() + y.value();

    result.node_ = std::make_shared<NodeAdd>(result.value());

    Var::createEdges_({x, y}, result);

    return result;
}

Var operator-(Var x, Var y)
{
    Var result = x.value() - y.value();

    result.node_ = std::make_shared<NodeSub>(result.value());

    Var::createEdges_({x, y}, result);

    return result;
}

Var operator-(Var x)
{
    Var result = -x.value();

    result.node_ = std::make_shared<NodeNeg>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var operator*(Var x, Var y)
{
    Var result = x.value() * y.value();

    result.node_ = std::make_shared<NodeMul>(result.value());

    Var::createEdges_({x, y}, result);

    return result;
}

Var operator/(Var x, Var y)
{
    Var result = x.value() / y.value();

    result.node_ = std::make_shared<NodeDiv>(result.value());

    Var::createEdges_({x, y}, result);

    return result;
}

std::ostream &operator<<(std::ostream &os, const Var &x)
{
    os << x.value() << " ± " << x.sigma();
    return os;
}

void Var::createEdges_(const std::initializer_list<Var> &inputNodes, Var &outputNode)
{
    for (const Var &x: inputNodes)
    {
        outputNode.children_.push_back(std::make_shared<Var>(x));
        outputNode.node_->children_.push_back(x.node_);
    }

    for (const std::shared_ptr<Var> &x: outputNode.children_)
    {
        x->parents_.push_back(std::make_shared<Var>(outputNode));
        x->node_->parents_.push_back(outputNode.node_);
    }
}

std::vector<Var> Var::sortedNodes_() const
{
    std::vector<Var> nodes;

    std::unordered_map<int, int> numParents;

    std::stack<Var> stack({*this});

    while (!stack.empty())
    {
        const Var node = stack.top();
        stack.pop();

        nodes.push_back(node);

        for (const std::shared_ptr<Var> &child: node.children_)
        {
            auto search = numParents.find(child->id());

            if (search != numParents.end())
            {
                --numParents[child->id()];
            }
            else
            {
                numParents[child->id()] = child->parents_.size() - 1;
            }

            if (numParents[child->id()] == 0)
            {
                stack.push(*child);
            }
        }
    }

    return nodes;
}

std::vector<Var> Var::inputNodes_() const
{
    const std::vector<Var> nodes = sortedNodes_();

    std::vector<Var> inputNodes;
    std::copy_if(nodes.begin(),
                 nodes.end(),
                 std::back_inserter(inputNodes),
                 [](const Var &node) { return node.children_.empty(); });

    return inputNodes;
}

double Var::covariance_(const Var &x, const Var &y)
{
    x.backprop();
    std::vector<Var> xNodes = x.inputNodes_();

    // Copy the gradients of the input nodes before backpropagating on the second graph
    std::unordered_map<int, double> xGradientMap;
    std::for_each(xNodes.begin(),
                  xNodes.end(),
                  [&xGradientMap](const Var &node) { xGradientMap.emplace(node.id(), node.derivative()); });

    y.backprop();
    const std::vector<Var> yNodes = y.inputNodes_();

    std::unordered_map<int, double> yGradientMap;
    std::for_each(yNodes.begin(),
                  yNodes.end(),
                  [&yGradientMap](const Var &node) { yGradientMap.emplace(node.id(), node.derivative()); });

    xNodes.insert(xNodes.end(), yNodes.begin(), yNodes.end());

    auto comparator = [](const Var &x, const Var &y) { return x.id() < y.id(); };
    std::set<Var, decltype(comparator)> nodes(xNodes.begin(), xNodes.end(), comparator);

    std::vector<double> xGradient, yGradient;

    for (const Var &node: nodes)
    {
        {
            auto search = xGradientMap.find(node.id());

            if (search != xGradientMap.end())
            {
                xGradient.push_back(search->second);
            }
            else
            {
                xGradient.push_back(0.0);
            }
        }

        {
            auto search = yGradientMap.find(node.id());

            if (search != yGradientMap.end())
            {
                yGradient.push_back(search->second);
            }
            else
            {
                yGradient.push_back(0.0);
            }
        }
    }

    std::vector<double> matrix;

    for (const Var &row: nodes)
    {
        for (const Var &col: nodes)
        {
            // This doesn't end up being a recursive call chain since `nodes` contains leaf variables
            matrix.push_back(covariance(row, col));
        }
    }

    std::vector<double> result = multiply(matrix, xGradient, nodes.size());
    result                     = multiply(yGradient, result, 1);

    return result[0];
}
}  // namespace cascade
