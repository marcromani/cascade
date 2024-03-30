#ifndef CASCADE_NODE_ACOSH_H
#define CASCADE_NODE_ACOSH_H

#include "../node.h"

namespace cascade
{
class NodeAcosh final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
