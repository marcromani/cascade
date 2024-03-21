#ifndef CASCADE_NODE_VAR_H
#define CASCADE_NODE_VAR_H

#include "node.h"

#include <memory>
#include <unordered_map>

namespace cascade
{
class NodeVar final : public Node
{
public:
    NodeVar();
    explicit NodeVar(double value);
    NodeVar(double value, double sigma);

    double sigma() const;

    static double covariance(const std::shared_ptr<NodeVar>&, const std::shared_ptr<NodeVar>&);
    static void setCovariance(std::shared_ptr<NodeVar>&, std::shared_ptr<NodeVar>&, double);

private:
    std::unordered_map<int, double> covariance_;

    void backprop_() override;
};
}  // namespace cascade

#endif
