#ifndef CASCADE_NODE_DIV_H
#define CASCADE_NODE_DIV_H

#include "node.h"

namespace cascade
{
class NodeDiv final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
