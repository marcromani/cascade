#ifndef CASCADE_NODE_ATANH_H
#define CASCADE_NODE_ATANH_H

#include "../node.h"

namespace cascade
{
class NodeAtanh final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
