#ifndef CASCADE_NODE_MUL_H
#define CASCADE_NODE_MUL_H

#include "node.h"

namespace cascade
{
class NodeMul final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
