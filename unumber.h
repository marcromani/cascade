#ifndef UNUMBER_H
#define UNUMBER_H

#include "unumber-base.h"
#include <ostream>

template <typename T>
class UNumber : public UNumberBase
{
public:
    /// @brief Constructs an instance with value 0 ± 0.
    UNumber();

    /// @brief Constructs an instance with value \p value ± 0.
    /// @param value
    UNumber(T value);

    /// @brief Constructs an instance with value \p value ± \p sigma.
    /// @param value
    /// @param sigma
    UNumber(T value, double sigma);

    /// @brief Copy constructor. Creates an instance with the same value as \p other but not correlated to it.
    /// @param other
    UNumber(const UNumber<T> &other);

    UNumber<T> &operator=(const UNumber<T> &other);

    // Accessors
    T value() const;
    double sigma() const;

    template <typename U>
    friend std::ostream &operator<<(std::ostream &, const UNumber<U> &);

private:
    T value_;
    double sigma_;
};

using UInt = UNumber<int>;
using UFloat = UNumber<float>;
using UDouble = UNumber<double>;

#include "unumber.tpp"

#endif
