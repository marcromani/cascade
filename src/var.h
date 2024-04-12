#ifndef CASCADE_VAR_H
#define CASCADE_VAR_H

#include <memory>
#include <ostream>
#include <vector>

namespace cascade
{
class Node;

class Var
{
public:
    /**
     * @brief Constructs an instance with value 0 ± 0.
     *
     */
    Var();

    /**
     * @brief Constructs an instance with value \p value ± 0.
     *
     * @param value
     */
    Var(double value);

    /**
     * @brief Constructs an instance with value \p value ± \p sigma.
     *
     * @param value
     * @param sigma
     */
    Var(double value, double sigma);

    /**
     * @brief Unique id that identifies a variable and tracks its correlations.
     *
     */
    int id() const;

    double value() const;
    double sigma() const;
    double derivative() const;

    bool setSigma(double);

    static double covariance(const Var &, const Var &);
    static bool setCovariance(const Var &, const Var &, double);

    void backprop() const;

    friend std::ostream &operator<<(std::ostream &, const Var &);

    /* Math operators */

    friend Var operator+(const Var &, const Var &);
    friend Var operator-(const Var &, const Var &);
    friend Var operator-(const Var &);
    friend Var operator*(const Var &, const Var &);
    friend Var operator/(const Var &, const Var &);

    /* Math functions */

    friend Var pow(const Var &, const Var &);

    friend Var sqrt(const Var &);

    friend Var abs(const Var &);

    friend Var exp(const Var &);
    friend Var exp2(const Var &);
    friend Var exp10(const Var &);

    friend Var log(const Var &);
    friend Var log2(const Var &);
    friend Var log10(const Var &);

    friend Var sin(const Var &);
    friend Var cos(const Var &);
    friend Var tan(const Var &);

    friend Var asin(const Var &);
    friend Var acos(const Var &);
    friend Var atan(const Var &);

    friend Var sinh(const Var &);
    friend Var cosh(const Var &);
    friend Var tanh(const Var &);

    friend Var asinh(const Var &);
    friend Var acosh(const Var &);
    friend Var atanh(const Var &);

    friend Var min(const Var &, const Var &);
    friend Var max(const Var &, const Var &);

private:
    static void createEdges(const std::initializer_list<Var> &inputNodes, Var &outputNode);

    /**
     * @brief Constructs a topologically sorted list of nodes of the implicit graph rooted at the caller instance.
     *
     * @return The sorted nodes.
     */
    std::vector<Var> sortedNodes() const;

    std::vector<Var> inputNodes() const;

    static double covariance_(const Var &, const Var &);

    std::shared_ptr<Node> node_;

    std::vector<std::shared_ptr<Var>> children_;
    std::vector<std::weak_ptr<Var>> parents_;
};
}  // namespace cascade

#endif
