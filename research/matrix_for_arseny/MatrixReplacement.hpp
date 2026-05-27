//
// Created by evgen on 25.03.2026.
//

#ifndef RESEARCH_MATRUXREPLACEMENT_HPP
#define RESEARCH_MATRUXREPLACEMENT_HPP

#include "types/Types.hpp"

#include <Eigen/Core>

namespace EMW::Math::LinAgl::Matrix::Wrappers {
/**
 * Класс-обёртка для факторизованный комплексной матрицы из двух матриц
 * @tparam MatrixType -- собственно та самая матрица
 */
template <typename MatrixType>
class VolumeMatrixReplacement : public Eigen::EigenBase<VolumeMatrixReplacement<MatrixType>> {
  public:
    // Required typedefs, constants, and method:
    typedef Types::complex_d Scalar;
    typedef Eigen::NumTraits<Scalar>::Real RealScalar;
    typedef Types::integer StorageIndex;
    constexpr static Types::integer Options_ = 0;

    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };

    decltype(auto) rows() const { return mat_.rows(); }
    decltype(auto) cols() const { return mat_.cols(); }

    template <typename Rhs>
    Eigen::Product<VolumeMatrixReplacement, Rhs, Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs> &x) const {
        return Eigen::Product<VolumeMatrixReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    // Custom API:
    VolumeMatrixReplacement() = default;

    VolumeMatrixReplacement(const MatrixType &mat, const Eigen::PermutationMatrix<Eigen::Dynamic> &perm)
        : mat_(mat), perm_mat(perm){};

    [[nodiscard]] const MatrixType &get_mat() const { return mat_; }
    [[nodiscard]] const Eigen::PermutationMatrix<Eigen::Dynamic>& get_perm() const { return perm_mat; }

  private:
    const MatrixType &mat_;
    const Eigen::PermutationMatrix<Eigen::Dynamic> &perm_mat;
};
} // namespace EMW::Math::LinAgl::Matrix::Wrappers

// Написание обёртки для умножения матрицы на вектор через специальную структуру внутри Eigen:
// VolumeMatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen::internal {
template <typename MatrixType>
using factored_matrix_replacement = EMW::Math::LinAgl::Matrix::Wrappers::VolumeMatrixReplacement<MatrixType>;

template <typename MatrixType1, typename Rhs>
struct generic_product_impl<factored_matrix_replacement<MatrixType1>, Rhs, SparseShape, DenseShape,
                            GemvProduct> // GEMV stands for matrix-vector
    : generic_product_impl_base<factored_matrix_replacement<MatrixType1>, Rhs,
                                generic_product_impl<factored_matrix_replacement<MatrixType1>, Rhs>> {
    typedef typename Product<factored_matrix_replacement<MatrixType1>, Rhs>::Scalar Scalar;

    template <typename Dest>
    static void scaleAndAddTo(Dest &dst, const factored_matrix_replacement<MatrixType1> &lhs, const Rhs &rhs,
                              const Scalar &alpha) {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
#if DEBUG
        assert(alpha == Scalar(1) && "scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);
#endif
            dst.noalias() += static_cast<Dest>(lhs.get_perm() * (lhs.get_mat() * (lhs.get_perm().transpose() * rhs)));
        }
    };
}

#endif //RESEARCH_MATRUXREPLACEMENT_HPP
