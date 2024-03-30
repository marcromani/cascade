#include "functions.h"

#include "../var.h"
#include "node_acos.h"
#include "node_acosh.h"
#include "node_asin.h"
#include "node_asinh.h"
#include "node_atan.h"
#include "node_atanh.h"
#include "node_cos.h"
#include "node_cosh.h"
#include "node_sin.h"
#include "node_sinh.h"
#include "node_tan.h"
#include "node_tanh.h"

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

Var tan(Var x)
{
    Var result = std::tan(x.value());

    result.node_ = std::make_shared<NodeTan>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var asin(Var x)
{
    Var result = std::asin(x.value());

    result.node_ = std::make_shared<NodeAsin>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var acos(Var x)
{
    Var result = std::acos(x.value());

    result.node_ = std::make_shared<NodeAcos>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var atan(Var x)
{
    Var result = std::atan(x.value());

    result.node_ = std::make_shared<NodeAtan>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var sinh(Var x)
{
    Var result = std::sinh(x.value());

    result.node_ = std::make_shared<NodeSinh>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var cosh(Var x)
{
    Var result = std::cosh(x.value());

    result.node_ = std::make_shared<NodeCosh>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var tanh(Var x)
{
    Var result = std::tanh(x.value());

    result.node_ = std::make_shared<NodeTanh>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var asinh(Var x)
{
    Var result = std::asinh(x.value());

    result.node_ = std::make_shared<NodeAsinh>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var acosh(Var x)
{
    Var result = std::acosh(x.value());

    result.node_ = std::make_shared<NodeAcosh>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var atanh(Var x)
{
    Var result = std::atanh(x.value());

    result.node_ = std::make_shared<NodeAtanh>(result.value());

    Var::createEdges_({x}, result);

    return result;
}
}  // namespace cascade
