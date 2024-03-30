#ifndef CASCADE_NODE_ABS_H
#define CASCADE_NODE_ABS_H

#include "../node.h"

namespace cascade
{
class NodeAbs final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
