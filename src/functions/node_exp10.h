#ifndef CASCADE_NODE_EXP10_H
#define CASCADE_NODE_EXP10_H

#include "../node.h"

namespace cascade
{
class NodeExp10 final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
