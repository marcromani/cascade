#ifndef NODE_H
#define NODE_H

#include <memory>
#include <vector>

namespace cascade
{
    class Node
    {
    public:
        Node();
        Node(double value);

        double value() const;
        double derivative() const;

        void setDerivative(double derivative);

    protected:
        virtual void backprop_();

        double value_;
        double derivative_;

        std::vector<std::shared_ptr<Node>> children_;
        std::vector<std::shared_ptr<Node>> parents_;

        friend class Var;
    };
}

#endif
