#ifndef CASCADE_TENSOR_DATA_H
#define CASCADE_TENSOR_DATA_H

#include <memory>
#include <vector>

#if CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace cascade
{
struct Tensor::TensorData final
{
    bool device;

    std::vector<size_t> shape;

    std::unique_ptr<float[]> hostData;
    std::unique_ptr<float[]> hostGrad;

#if CUDA_ENABLED
    bool hostDataNeedsUpdate;
    bool deviceDataNeedsUpdate;

    struct CudaDeleter final
    {
        void operator()(float *ptr) const { cudaFree(ptr); }
    };

    std::unique_ptr<float[], CudaDeleter> deviceData;
    std::unique_ptr<float[], CudaDeleter> deviceGrad;
#endif

    std::vector<Tensor> children;
    std::vector<Tensor> parents;
};
}  // namespace cascade

#endif
