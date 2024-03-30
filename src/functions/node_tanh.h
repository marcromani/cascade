#ifndef CASCADE_NODE_TANH_H
#define CASCADE_NODE_TANH_H

#include "../node.h"

namespace cascade
{
class NodeTanh final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
