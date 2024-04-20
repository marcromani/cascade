#ifndef CASCADE_NODE_NEG_H
#define CASCADE_NODE_NEG_H

#include "../node.h"

namespace cascade
{
class NodeNeg final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
