#ifndef VAR_H
#define VAR_H

#include "node.h"
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
        /**
         * @brief Constructs an instance with value 0 ± 0.
         * 
         */
        Var();

        /**
         * @brief Constructs an instance with value \p mean ± 0.
         * 
         * @param mean
         */
        Var(double mean);

        /**
         * @brief Constructs an instance with value \p mean ± \p sigma.
         * 
         * @param mean
         * @param sigma
         */
        Var(double mean, double sigma);

        /**
         * @brief Creates a deep copy, however, the new instance shares the covariance matrix with \p other,
         * and they both have the same \ref id.
         * 
         * @param other
         */
        Var(const Var &other);

        /**
         * @brief Creates a deep copy, however, the new instance shares the covariance matrix with \p other,
         * and they both have the same \ref id.
         * 
         * @param other
         */
        Var &operator=(const Var &other);

        double mean() const;
        double sigma() const;

        /**
         * @brief Unique index (up to copies) that identifies a variable and tracks its correlations.
         * 
         */
        int id() const;

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
        static void createEdges_(const std::initializer_list<Var> &inputNodes, Var &outputNode);

        /**
         * @brief Topological node sort. Allows \ref backprop_ to be called on the graph nodes while respecting edge dependencies.
         * 
         * @return The sorted nodes.
         */
        std::vector<Var> sortNodes_() const;

        std::shared_ptr<Node> node_;

        double sigma_;

        int index_;

        std::shared_ptr<std::unordered_map<int, double>> covariance_;

        std::vector<Var> children_;
        std::vector<Var> parents_;

        virtual void backprop_();

        static int counter_;
    };
}

#endif
