#ifndef NODE_H
#define NODE_H

#include <memory>
#include <vector>

namespace cascade
{
    class Node
    {
        friend class Var;

    public:
        Node();
        Node(double value);
        Node(double value, double sigma);

        int id() const;

        double value() const;
        double sigma() const;
        double derivative() const;

        void setDerivative(double derivative);

    protected:
        virtual void backprop_();

        int id_;

        double value_;
        double sigma_;
        double derivative_;

        std::vector<std::shared_ptr<Node>> children_;
        std::vector<std::shared_ptr<Node>> parents_;

    private:
        static int counter_;
    };
}

#endif
