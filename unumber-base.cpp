#include "unumber-base.h"

int UNumberBase::counter_ = 0;

UNumberBase::UNumberBase() : index_(counter_)
{
    ++counter_;
}

int UNumberBase::index() const
{
    return index_;
}

void setCovariance(UNumberBase &x, UNumberBase &y, double value)
{
    if (x.index_ < y.index_)
    {
        x.covariance_.emplace(y.index_, value);
    }
    else
    {
        y.covariance_.emplace(x.index_, value);
    }
}

double covariance(const UNumberBase &x, const UNumberBase &y)
{
    const UNumberBase *parent;
    const UNumberBase *child;

    if (x.index_ < y.index_)
    {
        parent = &x;
        child = &y;
    }
    else
    {
        parent = &y;
        child = &x;
    }

    auto search = parent->covariance_.find(child->index_);

    if (search != parent->covariance_.end())
    {
        return search->second;
    }
    else
    {
        return 0.0;
    }
}
