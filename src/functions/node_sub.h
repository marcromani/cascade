#ifndef CASCADE_NODE_SUB_H
#define CASCADE_NODE_SUB_H

#include "../node.h"

namespace cascade
{
class NodeSub final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
