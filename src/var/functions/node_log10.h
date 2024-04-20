#ifndef CASCADE_NODE_LOG10_H
#define CASCADE_NODE_LOG10_H

#include "../node.h"

namespace cascade
{
class NodeLog10 final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
