#ifndef CASCADE_NODE_SIN_H
#define CASCADE_NODE_SIN_H

#include "../node.h"

namespace cascade
{
class NodeSin final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
