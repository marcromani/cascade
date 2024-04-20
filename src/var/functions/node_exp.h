#ifndef CASCADE_NODE_EXP_H
#define CASCADE_NODE_EXP_H

#include "../node.h"

namespace cascade
{
class NodeExp final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
