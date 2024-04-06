#ifndef CASCADE_FUNCTIONS_H
#define CASCADE_FUNCTIONS_H

#include "var.h"

namespace cascade
{
Var pow(const Var&, const Var&);

Var sqrt(const Var&);

Var abs(const Var&);

Var exp(const Var&);
Var exp2(const Var&);
Var exp10(const Var&);

Var log(const Var&);
Var log2(const Var&);
Var log10(const Var&);

Var sin(const Var&);
Var cos(const Var&);
Var tan(const Var&);

Var asin(const Var&);
Var acos(const Var&);
Var atan(const Var&);

Var sinh(const Var&);
Var cosh(const Var&);
Var tanh(const Var&);

Var asinh(const Var&);
Var acosh(const Var&);
Var atanh(const Var&);

Var min(const Var&, const Var&);
Var max(const Var&, const Var&);
}  // namespace cascade

#endif
