#ifndef NODE_SIN
#define NODE_SIN

#include "node.h"

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
