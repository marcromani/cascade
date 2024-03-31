#ifndef CASCADE_NODE_MIN_H
#define CASCADE_NODE_MIN_H

#include "../node.h"

namespace cascade
{
class NodeMin final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
