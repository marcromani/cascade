#ifndef VAR_H
#define VAR_H

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
    double sigma();
    double derivative() const;

    static double covariance(Var &, Var &);
    static bool setCovariance(Var &, Var &, double);

    void backprop();

    // Math operators
    friend Var operator+(Var, Var);
    friend Var operator-(Var, Var);
    friend Var operator*(Var, Var);
    friend Var operator/(Var, Var);

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

    static double covariance_(Var &, Var &);

    std::shared_ptr<Node> node_;

    std::vector<Var> children_;
    std::vector<Var> parents_;
};
}  // namespace cascade

#endif
