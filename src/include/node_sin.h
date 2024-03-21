#ifndef CASCADE_NODE_SIN
#define CASCADE_NODE_SIN

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
