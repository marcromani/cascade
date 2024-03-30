#ifndef CASCADE_NODE_LOG2_H
#define CASCADE_NODE_LOG2_H

#include "../node.h"

namespace cascade
{
class NodeLog2 final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
