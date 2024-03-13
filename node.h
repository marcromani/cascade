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

    int id() const;

    double value() const;

    double derivative() const;
    void setDerivative(double derivative);

protected:
    static int counter_;

    int id_;

    double value_;
    double derivative_;

    std::vector<std::shared_ptr<Node>> children_;
    std::vector<std::shared_ptr<Node>> parents_;

private:
    virtual void backprop_() = 0;
};
}  // namespace cascade

#endif
