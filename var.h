#ifndef VAR_H
#define VAR_H

#include <functional>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

namespace cascade
{
    class Var
    {
    public:
        /// @brief Constructs an instance with value 0 ± 0.
        Var();

        /// @brief Constructs an instance with value \p mean ± 0.
        /// @param mean
        Var(double mean);

        /// @brief Constructs an instance with value \p mean ± \p sigma.
        /// @param mean
        /// @param sigma
        Var(double mean, double sigma);

        /// @brief Copy constructor. Creates a shallow copy, the new instance shares the internal state with \p other.
        /// @param other
        Var(const Var &other);

        /// @brief  Assignment operator. Creates a shallow copy, the new instance shares the internal state with \p other.
        /// @param other
        /// @return The shallow copy.
        Var &operator=(const Var &other);

        double mean() const;
        double sigma() const;

        int index() const;

        static void setCovariance(Var &, Var &, double);
        static double covariance(const Var &, const Var &);

        double derivative() const;
        void backprop();

        // Math operators
        friend Var operator+(Var, Var);
        friend Var operator-(Var, Var);
        friend Var operator*(Var, Var);
        friend Var operator/(Var, Var);

        friend std::ostream &operator<<(std::ostream &, const Var &);

    private:
        static void createEdges_(const std::initializer_list<Var> inputNodes, const Var &outputNode);

        std::vector<Var> sortNodes_() const;

        std::shared_ptr<double> mean_;
        std::shared_ptr<double> sigma_;

        std::shared_ptr<int> index_;

        std::shared_ptr<std::unordered_map<int, double>> covariance_;

        std::shared_ptr<std::vector<Var>> children_;
        std::shared_ptr<std::vector<Var>> parents_;

        std::shared_ptr<double> derivative_;
        std::function<void()> backprop_;

        static int counter_;
    };
}

#endif
