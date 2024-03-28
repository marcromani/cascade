#ifndef CASCADE_NODE_TAN
#define CASCADE_NODE_TAN

#include "node.h"

namespace cascade
{
class NodeTan final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
