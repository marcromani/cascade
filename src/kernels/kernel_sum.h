#ifndef CASCADE_KERNEL_SUM_H
#define CASCADE_KERNEL_SUM_H

#include <cstddef>

void kernelSumForward(float *result, const float *x, const float *y, size_t size);
void kernelSumBackward(float *x, float *y, size_t size, const size_t *shape);

#endif
