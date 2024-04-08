#ifndef CASCADE_TENSOR_H
#define CASCADE_TENSOR_H

#include <cstddef>
#include <vector>

class Tensor final
{
public:
    Tensor();
    explicit Tensor(const std::vector<size_t> &shape);
    explicit Tensor(const std::vector<size_t> &shape, const std::vector<float> &data);

    ~Tensor();

    size_t size() const;
    const std::vector<size_t> &shape() const;

    float *data();

    float &operator()(const std::vector<size_t> &indices);
    const float &operator()(const std::vector<size_t> &indices) const;

    Tensor operator+(const Tensor &other) const;

private:
    size_t index(const std::vector<size_t> &indices) const;

    void allocateMemory(size_t size);
    void freeMemory();

    void setData(const std::vector<float> &data);

    void elementwiseSumCPU(float *result, const float *a, const float *b, size_t size) const;
    void elementwiseSumGPU(float *result, const float *a, const float *b, size_t size) const;

private:
    std::vector<size_t> shape_;
    float *data_;
};

#endif
