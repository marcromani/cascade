#ifndef NODE_ADD_H
#define NODE_ADD_H

#include "node.h"

namespace cascade
{
    class NodeAdd final : public Node
    {
    public:
        using Node::Node;

    protected:
        void backprop_();
    };
}

#endif
