#ifndef CASCADE_NODE_SINH_H
#define CASCADE_NODE_SINH_H

#include "../node.h"

namespace cascade
{
class NodeSinh final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
