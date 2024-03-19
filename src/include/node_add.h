#ifndef NODE_ADD_H
#define NODE_ADD_H

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
