#ifndef UNUMBER_H
#define UNUMBER_H

#include "unumber-base.h"
#include <ostream>
#include <type_traits>

template <typename T>
class UNumber : public UNumberBase
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");

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
    template <typename U>
    UNumber(const UNumber<U> &other);

    template <typename U>
    UNumber<T> &operator=(const UNumber<U> &other);

    // Accessors
    T value() const;
    double sigma() const;

    // Math operators
    template <typename U, typename V>
    friend UNumber<double> operator+(const UNumber<U> &x, const UNumber<V> &y);

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
