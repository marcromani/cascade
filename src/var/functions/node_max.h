#ifndef CASCADE_NODE_MAX_H
#define CASCADE_NODE_MAX_H

#include "../node.h"

namespace cascade
{
class NodeMax final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
