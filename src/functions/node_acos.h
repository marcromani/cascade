#ifndef CASCADE_NODE_ACOS_H
#define CASCADE_NODE_ACOS_H

#include "../node.h"

namespace cascade
{
class NodeAcos final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
