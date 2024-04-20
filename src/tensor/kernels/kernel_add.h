#ifndef CASCADE_KERNEL_ADD_H
#define CASCADE_KERNEL_ADD_H

class Tensor;
namespace cascade
{
void kernelAddForward(const Tensor& result, const Tensor& x, const Tensor& y);
void kernelAddBackward(const Tensor& x, const Tensor& y);
}  // namespace cascade

#endif
