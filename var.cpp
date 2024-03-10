#include "var.h"
#include <stack>

namespace cascade
{
    int Var::counter_ = 0;

    Var::Var() : Var(0.0) {}

    Var::Var(double mean) : Var(mean, 0.0) {}

    Var::Var(double mean, double sigma) : mean_(mean), sigma_(sigma), index_(counter_), derivative_(0.0)
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
        mean_ = other.mean_;
        sigma_ = other.sigma_;

        index_ = other.index_;

        covariance_ = other.covariance_;

        children_ = other.children_;
        parents_ = other.parents_;

        derivative_ = other.derivative_;
        backprop_ = other.backprop_;

        return *this;
    }

    double Var::mean() const
    {
        return mean_;
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
        if (x.id() < y.id())
        {
            x.covariance_->emplace(y.id(), value);
        }
        else
        {
            y.covariance_->emplace(x.id(), value);
        }
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

    double Var::derivative() const
    {
        return derivative_;
    }

    void Var::backprop()
    {
        std::vector<Var> nodes = sortNodes_();

        for (Var &node : nodes)
        {
            node.derivative_ = 0.0;
        }

        derivative_ = 1.0;

        for (const Var &node : nodes)
        {
            if (node.backprop_)
            {
                node.backprop_();
            }
        }
    }

    Var operator+(Var x, Var y)
    {
        Var result = x.mean() + y.mean();

        Var::createEdges_({x, y}, result);

        result.backprop_ = [result]() {
            Var x = result.children_.at(0);
            Var y = result.children_.at(1);

            x.derivative_ += result.derivative_;
            y.derivative_ += result.derivative_;
        };

        return result;
    }

    Var operator-(Var x, Var y)
    {
        Var result = x.mean() - y.mean();

        Var::createEdges_({x, y}, result);

        result.backprop_ = [result]() {
            Var x = result.children_.at(0);
            Var y = result.children_.at(1);

            x.derivative_ += result.derivative_;
            y.derivative_ -= result.derivative_;
        };

        return result;
    }

    Var operator*(Var x, Var y)
    {
        Var result = x.mean() * y.mean();

        Var::createEdges_({x, y}, result);

        result.backprop_ = [result]() {
            Var x = result.children_.at(0);
            Var y = result.children_.at(1);

            x.derivative_ += y.mean() * result.derivative_;
            y.derivative_ += x.mean() * result.derivative_;
        };

        return result;
    }

    Var operator/(Var x, Var y)
    {
        Var result = x.mean() / y.mean();

        Var::createEdges_({x, y}, result);

        result.backprop_ = [result]() {
            Var x = result.children_.at(0);
            Var y = result.children_.at(1);

            x.derivative_ += (1 / y.mean()) * result.derivative_;
            y.derivative_ -= (x.mean() / (y.mean() * y.mean())) * result.derivative_;
        };

        return result;
    }

    std::ostream &operator<<(std::ostream &os, const Var &x)
    {
        os << x.mean() << " ± " << x.sigma();
        return os;
    }

    void Var::createEdges_(const std::initializer_list<Var> &inputNodes, Var &outputNode)
    {
        for (Var x : inputNodes)
        {
            outputNode.children_.push_back(x);
            x.parents_.push_back(outputNode);
        }
    }

    std::vector<Var> Var::sortNodes_() const
    {
        std::vector<Var> nodes;

        std::unordered_map<int, int> numParents;

        std::stack<Var> stack({*this});

        while (!stack.empty())
        {
            const Var &node = stack.top();
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
