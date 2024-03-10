#include "var.h"
#include <stack>

namespace cascade
{
    int Var::counter_ = 0;

    Var::Var() : Var(0.0) {}

    Var::Var(double mean) : Var(mean, 0.0) {}

    Var::Var(double mean, double sigma)
    {
        mean_ = std::make_shared<double>(mean);
        sigma_ = std::make_shared<double>(sigma);

        index_ = std::make_shared<int>(counter_);
        ++counter_;

        covariance_ = std::make_shared<std::unordered_map<int, double>>();

        children_ = std::make_shared<std::vector<Var>>();
        parents_ = std::make_shared<std::vector<Var>>();

        derivative_ = std::make_shared<double>(0);
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
        return *mean_;
    }

    double Var::sigma() const
    {
        return *sigma_;
    }

    int Var::index() const
    {
        return *index_;
    }

    void Var::setCovariance(Var &x, Var &y, double value)
    {
        if (x.index() < y.index())
        {
            x.covariance_->emplace(y.index(), value);
        }
        else
        {
            y.covariance_->emplace(x.index(), value);
        }
    }

    double Var::covariance(const Var &x, const Var &y)
    {
        const Var *parent;
        const Var *child;

        if (x.index() < y.index())
        {
            parent = &x;
            child = &y;
        }
        else
        {
            parent = &y;
            child = &x;
        }

        auto search = parent->covariance_->find(child->index());

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
        return *derivative_;
    }

    void Var::backprop()
    {
        const std::vector<Var> &nodes = sortNodes_();

        for (const Var &node : nodes)
        {
            *node.derivative_ = 0;
        }

        *derivative_ = 1;

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

        result.backprop_ = [&result]() {
            const Var &x = result.children_->at(0);
            const Var &y = result.children_->at(1);

            *x.derivative_ += *result.derivative_;
            *y.derivative_ += *result.derivative_;
        };

        return result;
    }

    Var operator-(Var x, Var y)
    {
        Var result = x.mean() - y.mean();

        Var::createEdges_({x, y}, result);

        result.backprop_ = [&result]() {
            const Var &x = result.children_->at(0);
            const Var &y = result.children_->at(1);

            *x.derivative_ += *result.derivative_;
            *y.derivative_ -= *result.derivative_;
        };

        return result;
    }

    Var operator*(Var x, Var y)
    {
        Var result = x.mean() * y.mean();

        Var::createEdges_({x, y}, result);

        result.backprop_ = [&result]() {
            const Var &x = result.children_->at(0);
            const Var &y = result.children_->at(1);

            *x.derivative_ += y.mean() * *result.derivative_;
            *y.derivative_ += x.mean() * *result.derivative_;
        };

        return result;
    }

    Var operator/(Var x, Var y)
    {
        Var result = x.mean() / y.mean();

        Var::createEdges_({x, y}, result);

        result.backprop_ = [&result]() {
            const Var &x = result.children_->at(0);
            const Var &y = result.children_->at(1);

            *x.derivative_ += (1 / y.mean()) * *result.derivative_;
            *y.derivative_ -= (x.mean() / (y.mean() * y.mean())) * *result.derivative_;
        };

        return result;
    }

    std::ostream &operator<<(std::ostream &os, const Var &x)
    {
        os << x.mean() << " ± " << x.sigma();
        return os;
    }

    void Var::createEdges_(const std::initializer_list<Var> inputNodes, const Var &outputNode)
    {
        for (const Var &x : inputNodes)
        {
            outputNode.children_->push_back(x);
            x.parents_->push_back(outputNode);
        }
    }

    // Topological node sort. Allows backprop_ to be called on the graph nodes while respecting edge dependencies.
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

            for (const Var &child : *node.children_)
            {
                auto search = numParents.find(child.index());

                if (search != numParents.end())
                {
                    --numParents[child.index()];
                }
                else
                {
                    numParents[child.index()] = child.parents_->size() - 1;
                }

                if (numParents[child.index()] == 0)
                {
                    stack.push(child);
                }
            }
        }

        return nodes;
    }
}
