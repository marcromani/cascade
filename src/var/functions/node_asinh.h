#ifndef CASCADE_NODE_ASINH_H
#define CASCADE_NODE_ASINH_H

#include "../node.h"

namespace cascade
{
class NodeAsinh final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
