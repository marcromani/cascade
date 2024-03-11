#include "var.h"
#include <stack>

#include "node-add.h"
#include "node-mul.h"

namespace cascade
{
    int Var::counter_ = 0;

    Var::Var() : Var(0.0) {}

    Var::Var(double mean) : Var(mean, 0.0) {}

    Var::Var(double mean, double sigma) : node_(new Node(mean)), sigma_(sigma), index_(counter_)
    {
        covariance_ = std::make_shared<std::unordered_map<int, double>>();
        covariance_->emplace(index_, sigma_ * sigma_);

        ++counter_;
    }

    Var::Var(const Var &other)
    {
        *this = other;
    }

    Var &Var::operator=(const Var &other)
    {
        node_ = other.node_;

        sigma_ = other.sigma_;

        index_ = other.index_;

        covariance_ = other.covariance_;

        children_ = other.children_;
        parents_ = other.parents_;

        return *this;
    }

    double Var::mean() const
    {
        return node_->value();
    }

    double Var::sigma() const
    {
        return sigma_;
    }

    int Var::id() const
    {
        return index_;
    }

    void Var::setCovariance(Var &x, Var &y, double value)
    {
        if (x.index_ < y.index_)
        {
            x.covariance_->emplace(y.index_, value);
        }
        else
        {
            y.covariance_->emplace(x.index_, value);
        }
    }

    double Var::covariance(const Var &x, const Var &y)
    {
        const Var *parent;
        const Var *child;

        if (x.index_ < y.index_)
        {
            parent = &x;
            child = &y;
        }
        else
        {
            parent = &y;
            child = &x;
        }

        auto search = parent->covariance_->find(child->index_);

        if (search != parent->covariance_->end())
        {
            return search->second;
        }
        else
        {
            return 0.0;
        }
    }

    double Var::derivative() const
    {
        return node_->derivative();
    }

    void Var::backprop()
    {
        std::vector<Var> nodes = sortNodes_();

        for (Var &node : nodes)
        {
            node.node_->derivative_ = 0.0;
        }

        node_->derivative_ = 1.0;

        for (Var &node : nodes)
        {
            node.backprop_();
        }
    }

    Var operator+(Var x, Var y)
    {
        Var result = x.mean() + y.mean();

        result.node_ = std::shared_ptr<Node>(new NodeAdd(result.mean()));

        Var::createEdges_({x, y}, result);

        return result;
    }

    // Var operator-(Var x, Var y)
    // {
    //     Var result = x.mean() - y.mean();

    //     result.node_ = std::shared_ptr<Node>(new NodeSub(result.mean()));

    //     Var::createEdges_({x, y}, result);

    //     return result;
    // }

    Var operator*(Var x, Var y)
    {
        Var result = x.mean() * y.mean();

        result.node_ = std::shared_ptr<Node>(new NodeMul(result.mean()));

        Var::createEdges_({x, y}, result);

        return result;
    }

    // Var operator/(Var x, Var y)
    // {
    //     Var result = x.mean() / y.mean();

    //     result.node_ = std::shared_ptr<Node>(new NodeDiv(result.mean()));

    //     Var::createEdges_({x, y}, result);

    //     return result;
    // }

    std::ostream &operator<<(std::ostream &os, const Var &x)
    {
        os << x.mean() << " ± " << x.sigma_;
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
                auto search = numParents.find(child.index_);

                if (search != numParents.end())
                {
                    --numParents[child.index_];
                }
                else
                {
                    numParents[child.index_] = child.parents_.size() - 1;
                }

                if (numParents[child.index_] == 0)
                {
                    stack.push(child);
                }
            }
        }

        return nodes;
    }

    void Var::backprop_()
    {
        node_->backprop_();
    }
}
