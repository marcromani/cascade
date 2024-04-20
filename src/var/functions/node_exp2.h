#ifndef CASCADE_NODE_EXP2_H
#define CASCADE_NODE_EXP2_H

#include "../node.h"

namespace cascade
{
class NodeExp2 final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
