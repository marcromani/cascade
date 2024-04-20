#ifndef CASCADE_NODE_POW_H
#define CASCADE_NODE_POW_H

#include "../node.h"

namespace cascade
{
class NodePow final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
