#ifndef CASCADE_NODE_ADD_H
#define CASCADE_NODE_ADD_H

#include "node.h"

namespace cascade
{
class NodeAdd final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
