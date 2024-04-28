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
    struct CudaDeleter final
    {
#if CUDA_ENABLED
        void operator()(float *ptr) const { cudaFree(ptr); }
#else
        void operator()(float *ptr) const { delete[] ptr; }
#endif
    };

    bool scalar;

    bool device;

    bool hostDataNeedsUpdate;
    bool deviceDataNeedsUpdate;

    std::unique_ptr<float[]> hostData;
    std::unique_ptr<float[]> hostGrad;

    std::unique_ptr<float[], CudaDeleter> deviceData;
    std::unique_ptr<float[], CudaDeleter> deviceGrad;

    std::vector<Tensor> children;
    std::vector<Tensor> parents;
};
}  // namespace cascade

#endif
