#include "var.h"
#include "node-add.h"
#include "node-mul.h"
#include <stack>

namespace cascade
{
    Var::Var() : Var(0.0) {}

    Var::Var(double value) : Var(value, 0.0) {}

    Var::Var(double value, double sigma) : node_(new Node(value, sigma))
    {
        covariance_ = std::make_shared<std::unordered_map<int, double>>();
        covariance_->emplace(node_->id_, sigma * sigma);
    }

    int Var::id() const
    {
        return node_->id_;
    }

    double Var::value() const
    {
        return node_->value_;
    }

    double Var::sigma() const
    {
        return node_->sigma_;
    }

    double Var::derivative() const
    {
        return node_->derivative_;
    }

    double Var::covariance(const Var &x, const Var &y)
    {
        const Var *parent;
        const Var *child;

        if (x.id() < y.id())
        {
            parent = &x;
            child = &y;
        }
        else
        {
            parent = &y;
            child = &x;
        }

        auto search = parent->covariance_->find(child->id());

        if (search != parent->covariance_->end())
        {
            return search->second;
        }
        else
        {
            return 0.0;
        }
    }

    void Var::setCovariance(Var &x, Var &y, double value)
    {
        if (x.id() < y.id())
        {
            x.covariance_->emplace(y.id(), value);
        }
        else
        {
            y.covariance_->emplace(x.id(), value);
        }
    }

    void Var::backprop()
    {
        std::vector<Var> nodes = sortNodes_();

        for (const Var &node : nodes)
        {
            node.node_->derivative_ = 0.0;
        }

        node_->derivative_ = 1.0;

        for (Var &node : nodes)
        {
            node.node_->backprop_();
        }
    }

    Var operator+(Var x, Var y)
    {
        Var result = x.value() + y.value();

        result.node_ = std::shared_ptr<Node>(new NodeAdd(result.value()));

        Var::createEdges_({x, y}, result);

        return result;
    }

    // Var operator-(Var x, Var y)
    // {
    //     Var result = x.value() - y.value();

    //     result.node_ = std::shared_ptr<Node>(new NodeSub(result.value()));

    //     Var::createEdges_({x, y}, result);

    //     return result;
    // }

    Var operator*(Var x, Var y)
    {
        Var result = x.value() * y.value();

        result.node_ = std::shared_ptr<Node>(new NodeMul(result.value()));

        Var::createEdges_({x, y}, result);

        return result;
    }

    // Var operator/(Var x, Var y)
    // {
    //     Var result = x.value() / y.value();

    //     result.node_ = std::shared_ptr<Node>(new NodeDiv(result.value()));

    //     Var::createEdges_({x, y}, result);

    //     return result;
    // }

    std::ostream &operator<<(std::ostream &os, const Var &x)
    {
        os << x.value() << " ± " << x.sigma();
        return os;
    }

    void Var::createEdges_(const std::initializer_list<Var> &inputNodes, Var &outputNode)
    {
        for (Var x : inputNodes)
        {
            outputNode.children_.push_back(x);
            outputNode.node_->children_.push_back(x.node_);
        }

        for (Var &x : outputNode.children_)
        {
            x.parents_.push_back(outputNode);
            x.node_->parents_.push_back(outputNode.node_);
        }
    }

    std::vector<Var> Var::sortNodes_() const
    {
        std::vector<Var> nodes;

        std::unordered_map<int, int> numParents;

        std::stack<Var> stack({*this});

        while (!stack.empty())
        {
            const Var node = stack.top();
            stack.pop();

            nodes.push_back(node);

            for (const Var &child : node.children_)
            {
                auto search = numParents.find(child.id());

                if (search != numParents.end())
                {
                    --numParents[child.id()];
                }
                else
                {
                    numParents[child.id()] = child.parents_.size() - 1;
                }

                if (numParents[child.id()] == 0)
                {
                    stack.push(child);
                }
            }
        }

        return nodes;
    }
}
