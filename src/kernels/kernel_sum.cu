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

__global__ void kernelSumBackward_(float *x, float *y, size_t size, const size_t *shape)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        const size_t numDims = shape[0];

        size_t indices[16];  // To avoid cudaMalloc, use a big enough number of dimensions

        size_t stride = 1;

        for (int i = numDims - 1; i >= 0; --i)
        {
            indices[i] = (idx / stride) % shape[i + 1];
            stride *= shape[i + 1];
        }

        size_t allEqual = 1;

        for (size_t i = 0; i < numDims / 2; ++i)
        {
            allEqual *= indices[i] == indices[2 * i];
        }

        if (allEqual)
        {
            x[idx] = 1.0;
            y[idx] = 1.0;
        }
        else
        {
            x[idx] = 0.0;
            y[idx] = 0.0;
        }
    }
}

void kernelSumBackward(float *x, float *y, size_t size, const size_t *shape)
{
    constexpr size_t blockSize = 256;
    const size_t numBlocks     = (size + blockSize - 1) / blockSize;

    kernelSumBackward_<<<numBlocks, blockSize>>>(x, y, size, shape);

    cudaDeviceSynchronize();
}
