#include "tensor.h"

// TODO: Can we remove the guards?
#ifdef __CUDACC__
#include <cuda_runtime.h>

__global__ void sumKernel(float *result, const float *a, const float *b, size_t size)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        result[idx] = a[idx] + b[idx];
    }
}

void Tensor::sumGPU(float *result, const float *a, const float *b, size_t size) const
{
    constexpr size_t blockSize = 256;
    const size_t numBlocks     = (size + blockSize - 1) / blockSize;

    sumKernel<<<numBlocks, blockSize>>>(result, a, b, size);

    cudaDeviceSynchronize();
}

#endif
