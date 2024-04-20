#ifndef CASCADE_NODE_SQRT_H
#define CASCADE_NODE_SQRT_H

#include "../node.h"

namespace cascade
{
class NodeSqrt final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
