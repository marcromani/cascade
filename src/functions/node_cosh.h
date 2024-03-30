#ifndef CASCADE_NODE_COSH_H
#define CASCADE_NODE_COSH_H

#include "../node.h"

namespace cascade
{
class NodeCosh final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
