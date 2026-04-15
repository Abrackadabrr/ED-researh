//
// Created by evgen on 07.08.2025.
//

#ifndef MATRIXPRECONDITIONED_HPP
#define MATRIXPRECONDITIONED_HPP

#include "math/matrix/iterative_solvers_coverage/IdentityPreconditioner.hpp"
#include "types/Types.hpp"
#include <Eigen/Core>

namespace Research::Matrix::Wrappers {

using namespace EMW;

/**
 * Класс-обёртка дествительной матрицы для работы с солверами внутри Eigen
 * @tparam MatrixType -- собственно та самая матрица
 */
template <typename MatrixType,
          typename PreconditionerType =
              Math::LinAgl::Matrix::Preconditioning::IdentityPreconditioner<Types::scalar, MatrixType>>
class MatrixReplacementReal : public Eigen::EigenBase<MatrixReplacementReal<MatrixType, PreconditionerType>> {
  public:
    // Required typedefs, constants, and method:
    typedef Types::scalar Scalar;
    typedef Eigen::NumTraits<Scalar>::Real RealScalar;
    typedef Types::integer StorageIndex;
    constexpr static Types::integer Options_ = 0;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };

    decltype(auto) rows() const { return mp_mat->rows(); }
    decltype(auto) cols() const { return mp_mat->cols(); }

    template <typename Rhs>
    Eigen::Product<MatrixReplacementReal, Rhs, Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs> &x) const {
        return Eigen::Product<MatrixReplacementReal, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    // Custom API:
    MatrixReplacementReal() = default;
    MatrixReplacementReal(const MatrixType &matrix, const Mesh::SurfaceMesh &mesh) : mp_mat(&matrix), precond(mesh) {}

    const MatrixType &my_matrix() const { return *mp_mat; }
    const PreconditionerType &get_preconditioner() const { return precond; }

  private:
    const MatrixType *mp_mat;
    const PreconditionerType precond;
};
} // namespace Research::Matrix::Wrappers

// Написание обёртки для умножения матрицы на вектор через специальную структуру внутри Eigen:
// MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen::internal {
template <typename MatrixType, typename PreconditionerType>
using matrix_replacement_real = Research::Matrix::Wrappers::MatrixReplacementReal<MatrixType, PreconditionerType>;

template <typename MatrixType, typename PreconditionerType, typename Rhs>
struct generic_product_impl<matrix_replacement_real<MatrixType, PreconditionerType>, Rhs, SparseShape, DenseShape,
                            GemvProduct> // GEMV stands for matrix-vector
    : generic_product_impl_base<matrix_replacement_real<MatrixType, PreconditionerType>, Rhs,
                                generic_product_impl<matrix_replacement_real<MatrixType, PreconditionerType>, Rhs>> {
    typedef typename Product<matrix_replacement_real<MatrixType, PreconditionerType>, Rhs>::Scalar Scalar;

    template <typename Dest>
    static void scaleAndAddTo(Dest &dst, const matrix_replacement_real<MatrixType, PreconditionerType> &lhs,
                              const Rhs &rhs, const Scalar &alpha) {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        const auto inter_result = lhs.my_matrix() * lhs.get_preconditioner().solve(rhs);
        dst.noalias() += alpha * inter_result;
    }
};
} // namespace Eigen::internal

namespace Research::Matrix::Wrappers {
/**
 * Класс-обёртка комплексной матрицы для работы с солверами внутри Eigen
 * @tparam MatrixType -- собственно та самая матрица
 */
template <typename MatrixType,
          typename PreconditionerType =
              Math::LinAgl::Matrix::Preconditioning::IdentityPreconditioner<Types::complex_d, MatrixType>>
class MatrixReplacementComplex : public Eigen::EigenBase<MatrixReplacementComplex<MatrixType, PreconditionerType>> {
  public:
    // Required typedefs, constants, and method:
    typedef Types::complex_d Scalar;
    typedef Eigen::NumTraits<Scalar>::Real RealScalar;
    typedef Types::integer StorageIndex;
    constexpr static Types::integer Options_ = 0;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };

    decltype(auto) rows() const { return mp_mat->rows(); }
    decltype(auto) cols() const { return mp_mat->cols(); }

    template <typename Rhs>
    Eigen::Product<MatrixReplacementComplex, Rhs, Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs> &x) const {
        return Eigen::Product<MatrixReplacementComplex, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    // Custom API:
    MatrixReplacementComplex() = default;
    MatrixReplacementComplex(const MatrixType &matrix, const MatrixType &direct_precond_inv)
        : mp_mat(&matrix), precond(direct_precond_inv) {}

    const MatrixType &my_matrix() const { return *mp_mat; }
    const PreconditionerType &get_preconditioner() const { return precond; }

  private:
    const MatrixType *mp_mat;
    const PreconditionerType precond;
};
} // namespace Research::Matrix::Wrappers

// Написание обёртки для умножения матрицы на вектор через специальную структуру внутри Eigen:
// MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen::internal {
template <typename MatrixType, typename PreconditionerType>
using matrix_replacement_complex = Research::Matrix::Wrappers::MatrixReplacementComplex<MatrixType, PreconditionerType>;

template <typename MatrixType, typename PreconditionerType, typename Rhs>
struct generic_product_impl<matrix_replacement_complex<MatrixType, PreconditionerType>, Rhs, SparseShape, DenseShape,
                            GemvProduct> // GEMV stands for matrix-vector
    : generic_product_impl_base<matrix_replacement_complex<MatrixType, PreconditionerType>, Rhs,
                                generic_product_impl<matrix_replacement_complex<MatrixType, PreconditionerType>, Rhs>> {
    typedef typename Product<matrix_replacement_complex<MatrixType, PreconditionerType>, Rhs>::Scalar Scalar;

    template <typename Dest>
    static void scaleAndAddTo(Dest &dst, const matrix_replacement_complex<MatrixType, PreconditionerType> &lhs,
                              const Rhs &rhs, const Scalar &alpha) {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        const auto inter_result = lhs.my_matrix() *  lhs.get_preconditioner().solve(rhs);
        dst.noalias() += alpha * inter_result;
    }
};
}


#endif //MATRIXPRECONDITIONED_HPP
