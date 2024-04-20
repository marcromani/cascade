#include "kernel_mul.h"
#include "tensor.h"
#include "tensor_data.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

namespace cascade
{
__global__ void kernelMulForward_(float *result, const float *x, const float *y, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        result[idx] = x[idx] * y[idx];
    }
}

void kernelMulForward(const Tensor &result, const Tensor &x, const Tensor &y)
{
    constexpr size_t blockSize = 256;

    size_t size = result.size();

    size_t numBlocks = (size + blockSize - 1) / blockSize;

    kernelMulForward_<<<numBlocks, blockSize>>>(
        result.data_->deviceData.get(), x.data_->deviceData.get(), y.data_->deviceData.get(), size);

    cudaDeviceSynchronize();
}

__global__ void kernelMulBackward_(float *dx,
                                   float *dy,
                                   const float *x,
                                   const float *y,
                                   size_t size,
                                   const size_t *shape,
                                   size_t dims)
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
            allEqual = allEqual && (indices[i] == indices[i + dims / 2]);
        }

        if (allEqual)
        {
            size_t childIdx = 0;
            stride          = 1;

            for (int i = dims / 2 - 1; i >= 0; --i)
            {
                childIdx += indices[i] * stride;
                stride *= shape[i];
            }

            dx[idx] = y[childIdx];
            dy[idx] = x[childIdx];
        }
        else
        {
            dx[idx] = 0.f;
            dy[idx] = 0.f;
        }
    }
}

void kernelMulBackward(const Tensor &x, const Tensor &y)
{
    size_t dims = x.shape().size();

    size_t *shapePtr;
    cudaMalloc(reinterpret_cast<void **>(&shapePtr), 2 * dims * sizeof(size_t));
    cudaMemcpy(shapePtr, x.shape().data(), dims * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(shapePtr + dims, x.shape().data(), dims * sizeof(size_t), cudaMemcpyHostToDevice);

    constexpr size_t blockSize = 256;

    size_t size = x.size();

    size_t numBlocks = (size * size + blockSize - 1) / blockSize;

    kernelMulBackward_<<<numBlocks, blockSize>>>(x.data_->deviceGrad.get(),
                                                 y.data_->deviceGrad.get(),
                                                 x.data_->deviceData.get(),
                                                 y.data_->deviceData.get(),
                                                 size * size,
                                                 shapePtr,
                                                 2 * dims);

    cudaDeviceSynchronize();

    cudaFree(shapePtr);
}
}  // namespace cascade
