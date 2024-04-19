#ifndef CASCADE_TENSOR_DATA_H
#define CASCADE_TENSOR_DATA_H

#include <memory>

#if CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace cascade
{
struct Tensor::TensorData final
{
    bool device;

    bool hostDataNeedsUpdate;
    bool deviceDataNeedsUpdate;

    std::unique_ptr<float[]> hostData;
    std::unique_ptr<float[]> hostGrad;

    std::unique_ptr<float[], CudaDeleter> deviceData;
    std::unique_ptr<float[], CudaDeleter> deviceGrad;
};
}  // namespace cascade

#endif
