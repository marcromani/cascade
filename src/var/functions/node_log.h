#ifndef CASCADE_NODE_LOG_H
#define CASCADE_NODE_LOG_H

#include "../node.h"

namespace cascade
{
class NodeLog final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
