#ifndef CASCADE_NODE_ATAN_H
#define CASCADE_NODE_ATAN_H

#include "../node.h"

namespace cascade
{
class NodeAtan final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
