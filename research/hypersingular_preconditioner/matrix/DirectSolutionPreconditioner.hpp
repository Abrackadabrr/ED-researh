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

namespace Research::Matrix::Preconditioning {

using namespace EMW;
/**
* Очень простой предобуславливатель
*/
template <typename scalar_t, typename matrix_t> class DirectPreconditioner {
    using vector_t = Types::VectorX<scalar_t>;
    // матрица, обратная к предобуславливателю
    const matrix_t& operator_S;

  public:
    explicit DirectPreconditioner(const matrix_t& S)
        : operator_S(S) {}

    vector_t solve(const vector_t &rhs) const {
        return operator_S * rhs;
    }
};

}

#endif //DIRECTSOLUTIONPRECONDITIONER_HPP
