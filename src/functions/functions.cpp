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
#include "node_exp.h"
#include "node_exp10.h"
#include "node_exp2.h"
#include "node_log.h"
#include "node_log10.h"
#include "node_log2.h"
#include "node_pow.h"
#include "node_sin.h"
#include "node_sinh.h"
#include "node_sqrt.h"
#include "node_tan.h"
#include "node_tanh.h"

#include <cmath>

namespace cascade
{
Var pow(Var x, Var y)
{
    Var result = std::pow(x.value(), y.value());

    result.node_ = std::make_shared<NodePow>(result.value());

    Var::createEdges_({x, y}, result);

    return result;
}

Var sqrt(Var x)
{
    Var result = std::sqrt(x.value());

    result.node_ = std::make_shared<NodeSqrt>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var exp(Var x)
{
    Var result = std::exp(x.value());

    result.node_ = std::make_shared<NodeExp>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var exp2(Var x)
{
    Var result = std::exp2(x.value());

    result.node_ = std::make_shared<NodeExp2>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var exp10(Var x)
{
    Var result = std::pow(10.0, x.value());

    result.node_ = std::make_shared<NodeExp10>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var log(Var x)
{
    Var result = std::log(x.value());

    result.node_ = std::make_shared<NodeLog>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var log2(Var x)
{
    Var result = std::log2(x.value());

    result.node_ = std::make_shared<NodeLog2>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

Var log10(Var x)
{
    Var result = std::log10(x.value());

    result.node_ = std::make_shared<NodeLog10>(result.value());

    Var::createEdges_({x}, result);

    return result;
}

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
