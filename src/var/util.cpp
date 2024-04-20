#include "util.h"

#include <vector>

namespace cascade
{
std::vector<double> multiply(const std::vector<double> &A, const std::vector<double> &B, int rowsA)
{
    const int colsA = A.size() / rowsA;
    const int colsB = B.size() / colsA;

    std::vector<double> result(rowsA * colsB, 0.0);

    for (int i = 0; i < rowsA; ++i)
    {
        for (int j = 0; j < colsB; ++j)
        {
            for (int k = 0; k < colsA; ++k)
            {
                result[colsB * i + j] += A[colsA * i + k] * B[colsB * k + j];
            }
        }
    }

    return result;
}
}  // namespace cascade
