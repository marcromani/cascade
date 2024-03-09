#ifndef UNUMBER_H
#error __FILE__ should only be included from unumber.h
#endif

template <typename T>
UNumber<T>::UNumber() : UNumber(0.0) {}

template <typename T>
UNumber<T>::UNumber(T value) : UNumber(value, 0.0) {}

template <typename T>
UNumber<T>::UNumber(T value, double sigma) : value_(value), sigma_(sigma) {}

template <typename T>
template <typename U>
UNumber<T>::UNumber(const UNumber<U> &other) : UNumber(T(other.value()), other.sigma())
{
}

template <typename T>
template <typename U>
UNumber<T> &UNumber<T>::operator=(const UNumber<U> &other)
{
    const UNumber<T> tmp = {T(other.value()), other.sigma()};

    index_ = tmp.index_;
    covariance_ = tmp.covariance_;

    value_ = tmp.value_;
    sigma_ = tmp.sigma_;

    return *this;
}

template <typename T>
T UNumber<T>::value() const
{
    return value_;
}

template <typename T>
double UNumber<T>::sigma() const
{
    return sigma_;
}

template <typename U, typename V>
UNumber<double> operator+(const UNumber<U> &x, const UNumber<V> &y)
{
    return {x.value_ + y.value_, 0.0};
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const UNumber<T> &x)
{
    os << x.value_ << " ± " << x.sigma_;
    return os;
}
