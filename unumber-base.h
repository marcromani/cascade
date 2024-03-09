#ifndef UNUMBER_BASE_H
#define UNUMBER_BASE_H

#include <unordered_map>

class UNumberBase
{
public:
    int index() const;

    friend void setCovariance(UNumberBase &, UNumberBase &, double);
    friend double covariance(const UNumberBase &, const UNumberBase &);

protected:
    UNumberBase();

    int index_;
    std::unordered_map<int, double> covariance_;

private:
    static int counter_;
};

#endif
