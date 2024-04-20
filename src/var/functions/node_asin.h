#ifndef CASCADE_NODE_ASIN_H
#define CASCADE_NODE_ASIN_H

#include "../node.h"

namespace cascade
{
class NodeAsin final : public Node
{
public:
    using Node::Node;

private:
    void backprop_() override;
};
}  // namespace cascade

#endif
