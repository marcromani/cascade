#include "node_cos.h"
#include "node_sin.h"
#include "var.h"

#include <cmath>

namespace cascade
{
Var sin(Var x)
{
    Var result = std::sin(x.value());

    result.node_ = std::make_shared<NodeSin>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var cos(Var x)
{
    Var result = std::cos(x.value());

    result.node_ = std::make_shared<NodeCos>(result.value());

    Var::createEdges_({x}, result);

    return result;
}
}  // namespace cascade
