#ifndef CASCADE_FUNCTIONS_H
#define CASCADE_FUNCTIONS_H

#include "../var.h"

namespace cascade
{
Var pow(Var, Var);

Var sqrt(Var);

Var exp(Var);
Var exp2(Var);
Var exp10(Var);

Var log(Var);
Var log2(Var);
Var log10(Var);

Var sin(Var);
Var cos(Var);
Var tan(Var);

Var asin(Var);
Var acos(Var);
Var atan(Var);

Var sinh(Var);
Var cosh(Var);
Var tanh(Var);

Var asinh(Var);
Var acosh(Var);
Var atanh(Var);
}  // namespace cascade

#endif
