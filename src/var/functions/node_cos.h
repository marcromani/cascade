#ifndef CASCADE_NODE_COS_H
#define CASCADE_NODE_COS_H

#include "../node.h"

namespace cascade
{
class NodeCos final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
