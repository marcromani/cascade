#ifndef CASCADE_NODE_TAN_H
#define CASCADE_NODE_TAN_H

#include "../node.h"

namespace cascade
{
class NodeTan final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
