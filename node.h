#ifndef NODE_H
#define NODE_H

#include <memory>
#include <unordered_map>
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

    static double covariance(const std::shared_ptr<Node>, const std::shared_ptr<Node>);
    static void setCovariance(std::shared_ptr<Node>, std::shared_ptr<Node>, double);

protected:
    int id_;

    double value_;
    double sigma_;
    double derivative_;

    std::unordered_map<int, double> covariance_;

    std::vector<std::shared_ptr<Node>> children_;
    std::vector<std::shared_ptr<Node>> parents_;

private:
    virtual void backprop_() = 0;

    static int counter_;
};
}  // namespace cascade

#endif
