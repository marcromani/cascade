#include "kernel_sum.h"

#include <cuda_runtime.h>

__global__ void sumKernel_(float *result, const float *a, const float *b, size_t size)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        result[idx] = a[idx] + b[idx];
    }
}

void sumKernel(float *result, const float *a, const float *b, size_t size)
{
    constexpr size_t blockSize = 256;
    const size_t numBlocks     = (size + blockSize - 1) / blockSize;

    sumKernel_<<<numBlocks, blockSize>>>(result, a, b, size);

    cudaDeviceSynchronize();
}
