#ifndef CASCADE_KERNEL_MUL_H
#define CASCADE_KERNEL_MUL_H

class Tensor;

namespace cascade
{
void kernelMulForward(const Tensor& result, const Tensor& x, const Tensor& y);
void kernelMulBackward(const Tensor& x, const Tensor& y);
}  // namespace cascade

#endif
