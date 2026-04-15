//
// Created by evgen on 04.09.2025.
//

#ifndef DIRECTSOLUTIONPRECONDITIONER_HPP
#define DIRECTSOLUTIONPRECONDITIONER_HPP

#include "MatrixTraits.hpp"

#include "types/Types.hpp"

#include "../OperatorS_simple.hpp"

#include "Eigen/SparseCore"

#include <iostream>
#include <research/hypersingular_preconditioner/DiscreteLaplacian.hpp>

namespace Research::Matrix::Preconditioning {

using namespace EMW;
/**
* Очень простой предобуславливатель
*/
template <typename scalar_t, typename matrix_t> class CustomPreconditioner {
    using vector_t = Types::VectorX<scalar_t>;
    // матрица, обратная к предобуславливателю
    const matrix_t& operator_S;
    const Eigen::SparseMatrix<scalar_t> L;
    constexpr static Types::scalar alpha = 0.1;
    constexpr static Types::scalar one_m_alpha = 1 - alpha;

  public:
    CustomPreconditioner(const matrix_t& S, Types::index N, Types::scalar h, Types::complex_d k)
        : operator_S(S), L(DiscreteLaplacian::discreteHelmholtz(k, N + 2, N + 2, h, h)) {}

    vector_t solve(const vector_t &rhs) const {
        const vector_t semires = operator_S * rhs;
        return alpha * semires + one_m_alpha  * (L * semires);
    }
};

}

#endif //DIRECTSOLUTIONPRECONDITIONER_HPP
