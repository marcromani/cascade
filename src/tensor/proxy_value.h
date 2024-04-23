#ifndef CASCADE_PROXY_VALUE_H
#define CASCADE_PROXY_VALUE_H

namespace cascade
{
class Tensor::ProxyValue final
{
public:
    ProxyValue(float &value, bool *update) : value_(value), update_(update) {}

    operator float() const { return value_; }

    float operator=(float value)
    {
        value_ = value;

        if (update_ != nullptr)
        {
            *update_ = true;
        }

        return value;
    }

private:
    float &value_;
    bool *update_;
};
}  // namespace cascade

#endif
