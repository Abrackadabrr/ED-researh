//
// Created by evgen on 30.03.2026.
//

#ifndef DIAGONALPRECONDITIONER_HPP
#define DIAGONALPRECONDITIONER_HPP

#include "types/Types.hpp"

namespace Research::VolDie
{
    template <typename scalar_t, typename matrix_t>
    class IdentityPreconditioner
    {
        using vector_t = EMW::Types::VectorX<scalar_t>;

    public:
        IdentityPreconditioner(const matrix_t& volume_op)
        {
        }
        IdentityPreconditioner(const matrix_t& volume_op, const vector_t& eps_minus_one)
        {
        }

        vector_t solve(const vector_t& rhs) const { return rhs; }
    };

    template <typename scalar_t, typename matrix_t>
    class DiagonalPreconditioner
    {
        using vector_t = EMW::Types::VectorX<scalar_t>;

        // Храним только обратные элементы к диагональным
        vector_t invdiag;
        bool inited = false;
        bool matrix_initialized = false;

    public:
        DiagonalPreconditioner(const matrix_t& volume_op): invdiag(volume_op.diagonal().cwiseInverse()),
                                                           matrix_initialized(true)
        {
        }

        /** ну тут проблема с делением на ноль но я забил */
        DiagonalPreconditioner(const matrix_t& volume_op, const vector_t& eps_minus_one) :
            invdiag(
                (volume_op.diagonal().cwiseProduct(eps_minus_one)).
                cwiseInverse())
        {
            size_t big_count = 0;
            for (size_t idx = 0; idx < invdiag.size(); ++idx)
            {
                if (std::isnan(std::norm(invdiag[idx])) || std::abs(invdiag[idx]) > 1e7)
                {
                    big_count += 1;
                    invdiag[idx] = 1;
                }
            }
            std::cout << "Big values in prec:" << big_count << std::endl;
        }

        void attach_epsilon_matrix(const vector_t& eps_minus_one)
        {
            if (!matrix_initialized || inited) {
                throw std::runtime_error("DiagonalPreconditioner: problems with diagonal preconditioner");
            }
            invdiag.cwiseProduct(eps_minus_one.cwiseInverse());
        }

        // Решается система уравнений Dx = rsh
        // D -- это наша диагонлаьная матрица
        vector_t solve(const vector_t& rhs) const { return rhs.cwiseProduct(invdiag); }
    };
}

#endif //DIAGONALPRECONDITIONER_HPP
