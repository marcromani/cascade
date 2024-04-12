#include "kernel_sum.h"

#include <cuda_runtime.h>

__global__ void kernelSumForward_(float *result, const float *x, const float *y, size_t size)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        result[idx] = x[idx] + y[idx];
    }
}

void kernelSumForward(float *result, const float *x, const float *y, size_t size)
{
    constexpr size_t blockSize = 256;
    const size_t numBlocks     = (size + blockSize - 1) / blockSize;

    kernelSumForward_<<<numBlocks, blockSize>>>(result, x, y, size);

    cudaDeviceSynchronize();
}

__global__ void kernelSumBackward_(float *result, const float *x, const float *y, size_t size)
{
    // const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // size_t indices[shape.size()];

    // size_t stride = 1;

    // for (int i = shape.size() - 1; i >= 0; --i)
    // {
    //     indices[i] = (idx / stride) % shape[i];
    //     stride *= shape[i];
    // }
}

void kernelSumBackward(float *result, const float *x, const float *y, size_t size) {}
