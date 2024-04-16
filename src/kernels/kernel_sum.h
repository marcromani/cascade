#ifndef CASCADE_KERNEL_SUM_H
#define CASCADE_KERNEL_SUM_H

class Tensor;
namespace cascade
{
void kernelSumForward(const Tensor& result, const Tensor& x, const Tensor& y);
void kernelSumBackward(const Tensor& x, const Tensor& y);
}  // namespace cascade

#endif
