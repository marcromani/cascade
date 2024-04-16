#include "kernel_sum.h"
#include "tensor.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

namespace cascade
{
__global__ void kernelSumForward_(float *result, const float *x, const float *y, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        result[idx] = x[idx] + y[idx];
    }
}

void kernelSumForward(const Tensor &result, const Tensor &x, const Tensor &y)
{
    constexpr size_t blockSize = 256;

    size_t size = result.size();

    size_t numBlocks = (size + blockSize - 1) / blockSize;

    kernelSumForward_<<<numBlocks, blockSize>>>(
        result.deviceData_.get(), x.deviceData_.get(), y.deviceData_.get(), size);

    cudaDeviceSynchronize();
}

__global__ void kernelSumBackward_(float *x, float *y, size_t size, const size_t *shape, size_t dims)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        size_t indices[1024];  // Big enough number of dimensions to avoid cudaMalloc

        size_t stride = 1;

        for (int i = dims - 1; i >= 0; --i)
        {
            indices[i] = (idx / stride) % shape[i];
            stride *= shape[i];
        }

        bool allEqual = true;

        for (size_t i = 0; (i < dims / 2) && allEqual; ++i)
        {
            allEqual = allEqual && (indices[i] == indices[2 * i]);
        }

        if (allEqual)
        {
            x[idx] = 1.f;
            y[idx] = 1.f;
        }
        else
        {
            x[idx] = 3.f;
            y[idx] = 30.f;
        }
    }
}

void kernelSumBackward(const Tensor &x, const Tensor &y)
{
    size_t dims = x.shape().size();

    size_t *shapePtr;
    cudaMalloc(reinterpret_cast<void **>(&shapePtr), 2 * dims * sizeof(size_t));
    cudaMemcpy(shapePtr, x.shape().data(), dims * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(shapePtr + dims, x.shape().data(), dims * sizeof(size_t), cudaMemcpyHostToDevice);

    constexpr size_t blockSize = 256;

    size_t size = x.size();

    size_t numBlocks = (size * size + blockSize - 1) / blockSize;

    kernelSumBackward_<<<numBlocks, blockSize>>>(
        x.deviceGrad_.get(), y.deviceGrad_.get(), size * size, shapePtr, 2 * dims);

    cudaDeviceSynchronize();

    cudaFree(shapePtr);
}
}  // namespace cascade
