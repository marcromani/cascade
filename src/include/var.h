#ifndef CASCADE_VAR_H
#define CASCADE_VAR_H

#include "node.h"

#include <memory>
#include <ostream>
#include <vector>

namespace cascade
{
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
    static bool setCovariance(Var &, Var &, double);

    void backprop() const;

    // Math operators
    friend Var operator+(Var, Var);
    friend Var operator-(Var, Var);
    friend Var operator*(Var, Var);
    friend Var operator/(Var, Var);

    // Unary functions
    friend Var sin(Var);
    friend Var cos(Var);
    friend Var tan(Var);

    friend std::ostream &operator<<(std::ostream &, Var &);

private:
    static void createEdges_(const std::initializer_list<Var> &inputNodes, Var &outputNode);

    /**
     * @brief Constructs a topologically sorted list of nodes of the implicit graph rooted at the caller instance.
     *
     * @return The sorted nodes.
     */
    std::vector<Var> sortedNodes_() const;

    std::vector<Var> inputNodes_() const;

    static double covariance_(const Var &, const Var &);

    std::shared_ptr<Node> node_;

    std::vector<Var> children_;
    std::vector<Var> parents_;
};
}  // namespace cascade

#endif
