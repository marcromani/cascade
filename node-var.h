#ifndef NODE_VAR_H
#define NODE_VAR_H

#include "node.h"

namespace cascade
{
class NodeVar final : public Node
{
public:
    using Node::Node;

private:
    void backprop_();
};
}  // namespace cascade

#endif
