//
// Created by evgen on 25.03.2026.
//

#ifndef RESEARCH_MATRUXREPLACEMENT_HPP
#define RESEARCH_MATRUXREPLACEMENT_HPP

#include "Preconditioning.hpp"
#include "types/Types.hpp"

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace EMW::Math::LinAgl::Matrix::Wrappers {
/**
 * Класс-обёртка для факторизованный комплексной матрицы из двух матриц
 * @tparam MatrixType -- собственно та самая матрица
 */
template <typename MatrixType1,
          typename PreconditionerType = Research::VolDie::IdentityPreconditioner<Types::complex_d, MatrixType1>>
class VolumeOperatorMatrixReplacement
    : public Eigen::EigenBase<VolumeOperatorMatrixReplacement<MatrixType1, PreconditionerType>> {
  public:
    // Required typedefs, constants, and method:
    typedef Types::complex_d Scalar;
    typedef Eigen::NumTraits<Scalar>::Real RealScalar;
    typedef Types::integer StorageIndex;
    constexpr static Types::integer Options_ = 0;
    constexpr static auto zeroLike = 1e-7;

    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };

    decltype(auto) rows() const { return mat_.rows(); }
    decltype(auto) cols() const { return mat_.cols(); }

    template <typename Rhs>
    Eigen::Product<VolumeOperatorMatrixReplacement, Rhs, Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs> &x) const {
        return Eigen::Product<VolumeOperatorMatrixReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    // Custom API:
    VolumeOperatorMatrixReplacement() = default;

    VolumeOperatorMatrixReplacement(const MatrixType1 &mat, const Types::VectorX<Scalar> &eps_vec)
        : mat_(mat), mask_(Types::VectorX<RealScalar>::Zero(mat_.rows())), precond_(mat_) {
        if (eps_vec.size() != mat.cols()) {
            throw std::invalid_argument("Epsilon is not set up correctly");
        }
        for (size_t i = 0; i < mat.rows(); ++i) {
            mask_[i] = (std::abs(eps_vec[i]) > zeroLike);
        }
        epsilon_vec_ = eps_vec.cwiseProduct(mask_);
        // Нули в epsilon_vec_ нужно заполнить единицами и отправить в диагональный предобуславливатель
        decltype(auto) epsilon_minus_one_for_preconditioner = epsilon_vec_;
        for (size_t i = 0; i < mat.rows(); ++i) {
            if (std::abs(epsilon_vec_[i]) < zeroLike)
                epsilon_minus_one_for_preconditioner[i] = 1;
        }
        // precond_.attach_epsilon_matrix(epsilon_minus_one_for_preconditioner);
    }

    const PreconditionerType &get_preconditioner() const { return precond_; }
    const MatrixType1 &get_mat() const { return mat_; }
    [[nodiscard]] const Types::VectorX<Scalar> &get_epsilon_vec() const { return epsilon_vec_; }
    [[nodiscard]] const Types::VectorX<RealScalar> &get_mask() const { return mask_; }

    // Изменение правой части согласно насчитанной маске
    //
    // Это нужно для того, чтобы занулить фиктивные компоненты в невязке, которые необходимы
    // только для устраивания дважды тёплицевой структуры.
    //
    template <typename Rhs> [[nodiscard]] Types::VectorX<Scalar> modify_rhs_according_to_mask(Rhs &&rhs) const {
        return std::forward<Rhs>(rhs).cwiseProduct(mask_);
    }

  private:
    const MatrixType1 &mat_;
    // Вектора, которые отвечают за диагональную матрицу epsilon - 1
    Types::VectorX<Scalar> epsilon_vec_;
    // Нужна для отсечения лишних элементов в векторе неизвестных (так как кое-где не нужно решать систему)
    Types::VectorX<RealScalar> mask_;
    PreconditionerType precond_;
};

/**
 * Matrix-free representation of the right-preconditioned volume equation
 *
 *     P (I - K D) P C^{-1} P y = P b,
 *
 * where D = diag(epsilon - 1), P removes cells outside the scatterer and C is
 * a full-grid right preconditioner. GMRES solves for y; recover_solution(y)
 * returns the physical coefficient vector x = P C^{-1} P y.
 */
template <typename MatrixType, typename RightPreconditionerType>
class RightPreconditionedVolumeOperatorMatrixReplacement
    : public Eigen::EigenBase<
          RightPreconditionedVolumeOperatorMatrixReplacement<MatrixType, RightPreconditionerType>> {
  public:
    using Scalar = Types::complex_d;
    using RealScalar = Eigen::NumTraits<Scalar>::Real;
    using StorageIndex = Types::integer;
    using Vector = Types::VectorX<Scalar>;

    constexpr static Types::integer Options_ = 0;
    constexpr static RealScalar zeroLike = 1e-7;

    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };

    [[nodiscard]] decltype(auto) rows() const { return mat_.rows(); }
    [[nodiscard]] decltype(auto) cols() const { return mat_.cols(); }

    template <typename Rhs>
    Eigen::Product<RightPreconditionedVolumeOperatorMatrixReplacement, Rhs, Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs> &x) const {
        return Eigen::Product<RightPreconditionedVolumeOperatorMatrixReplacement, Rhs, Eigen::AliasFreeProduct>(
            *this, x.derived());
    }

    RightPreconditionedVolumeOperatorMatrixReplacement(const MatrixType &mat, const Vector &epsilon_vec,
                                                        const RightPreconditionerType &right_preconditioner)
        : mat_(mat), right_preconditioner_(right_preconditioner), mask_(Types::VectorX<RealScalar>::Zero(mat.rows())) {
        if (mat.rows() != mat.cols() || epsilon_vec.size() != mat.cols())
            throw std::invalid_argument("Epsilon is not set up correctly");
        if (right_preconditioner.rows() != mat.rows() || right_preconditioner.cols() != mat.cols())
            throw std::invalid_argument("The right preconditioner has incompatible dimensions");

        for (Eigen::Index index = 0; index < epsilon_vec.size(); ++index)
            mask_(index) = std::abs(epsilon_vec(index)) > zeroLike ? RealScalar{1} : RealScalar{0};
        epsilon_vec_ = epsilon_vec.cwiseProduct(mask_);
    }

    [[nodiscard]] const MatrixType &get_mat() const { return mat_; }
    [[nodiscard]] const RightPreconditionerType &get_right_preconditioner() const {
        return right_preconditioner_;
    }
    [[nodiscard]] const Vector &get_epsilon_vec() const { return epsilon_vec_; }
    [[nodiscard]] const Types::VectorX<RealScalar> &get_mask() const { return mask_; }

    template <typename Rhs> [[nodiscard]] Vector modify_rhs_according_to_mask(const Rhs &rhs) const {
        if (rhs.size() != rows())
            throw std::invalid_argument("The right-hand side has incompatible dimensions");
        return rhs.cwiseProduct(mask_);
    }

    /** Applies R = P C^{-1} P to a GMRES vector through the FFT preconditioner. */
    template <typename Rhs> [[nodiscard]] Vector apply_right_preconditioner(const Rhs &rhs) const {
        if (rhs.size() != cols())
            throw std::invalid_argument("The vector has incompatible dimensions");
        const Vector projected_rhs = rhs.cwiseProduct(mask_);
        return right_preconditioner_.solve(projected_rhs).cwiseProduct(mask_);
    }

    /** Converts the transformed GMRES unknown y to the physical unknown x. */
    template <typename Rhs> [[nodiscard]] Vector recover_solution(const Rhs &transformed_solution) const {
        return apply_right_preconditioner(transformed_solution);
    }

  private:
    const MatrixType &mat_;
    const RightPreconditionerType &right_preconditioner_;
    Vector epsilon_vec_;
    Types::VectorX<RealScalar> mask_;
};

template<typename matrix_t>
class SimpleVolumeOperator
    : public Eigen::EigenBase<SimpleVolumeOperator<matrix_t>> {
  public:
    // Required typedefs, constants, and method:
    typedef Types::complex_d Scalar;
    typedef Eigen::NumTraits<Scalar>::Real RealScalar;
    typedef Types::integer StorageIndex;
    constexpr static Types::integer Options_ = 0;
    constexpr static auto zeroLike = 1e-7;

    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };

    decltype(auto) rows() const { return mat_.rows(); }
    decltype(auto) cols() const { return mat_.cols(); }

    template <typename Rhs>
    Eigen::Product<SimpleVolumeOperator, Rhs, Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs> &x) const {
        return Eigen::Product<SimpleVolumeOperator, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    // Custom API:
    SimpleVolumeOperator() = default;

    template<typename vector_t>
    SimpleVolumeOperator(const matrix_t &mat, vector_t&& eps_vec)
        : mat_(mat), epsilon_vec_(eps_vec) {
        if (eps_vec.size() != mat.cols()) {
            throw std::invalid_argument("Epsilon is not set up correctly");
        }
    }

    const matrix_t &get_mat() const { return mat_; }
    [[nodiscard]] const Types::VectorX<Scalar> &get_epsilon_vec() const { return epsilon_vec_; }

  private:
    const matrix_t &mat_;
    // Вектора, которые отвечают за диагональную матрицу epsilon - 1
    Types::VectorX<Scalar> epsilon_vec_;
};
} // namespace EMW::Math::LinAgl::Matrix::Wrappers

// Написание обёртки для умножения матрицы на вектор через специальную структуру внутри Eigen:
// MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen::internal {
// ----------- Hard replacement ------------
template <typename MatrixType1, typename PreconditionerType>
using factored_matrix_replacement =
    EMW::Math::LinAgl::Matrix::Wrappers::VolumeOperatorMatrixReplacement<MatrixType1, PreconditionerType>;

template <typename MatrixType1, typename PreconditionerType, typename Rhs>
struct generic_product_impl<factored_matrix_replacement<MatrixType1, PreconditionerType>, Rhs, SparseShape, DenseShape,
                            GemvProduct> // GEMV stands for matrix-vector
    : generic_product_impl_base<
          factored_matrix_replacement<MatrixType1, PreconditionerType>, Rhs,
          generic_product_impl<factored_matrix_replacement<MatrixType1, PreconditionerType>, Rhs>> {
    typedef typename Product<factored_matrix_replacement<MatrixType1, PreconditionerType>, Rhs>::Scalar Scalar;

    template <typename Dest>
    static void scaleAndAddTo(Dest &dst, const factored_matrix_replacement<MatrixType1, PreconditionerType> &lhs,
                              const Rhs &rhs, const Scalar &alpha) {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        // Для оператора объемного уравнения пишем dst += (I - K * eps) P^{-1} * rhs
#if DEBUG
        assert(alpha == Scalar(1) && "scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);
#endif
            const auto& mask = lhs.get_mask();
            Rhs preconditioned = lhs.get_preconditioner().solve(rhs).cwiseProduct(mask);
//            VectorX<Scalar> tmp = preconditioned.cwiseProduct(lhs.get_epsilon_vec());
            dst.noalias() += preconditioned - (lhs.get_mat() * preconditioned.cwiseProduct(lhs.get_epsilon_vec())).cwiseProduct(mask);
        }
    };

// -------- Right-preconditioned volume operator --------
template <typename MatrixType, typename RightPreconditionerType>
using right_preconditioned_volume_operator =
    EMW::Math::LinAgl::Matrix::Wrappers::RightPreconditionedVolumeOperatorMatrixReplacement<
        MatrixType, RightPreconditionerType>;

template <typename MatrixType, typename RightPreconditionerType, typename Rhs>
struct generic_product_impl<right_preconditioned_volume_operator<MatrixType, RightPreconditionerType>, Rhs,
                            SparseShape, DenseShape, GemvProduct>
    : generic_product_impl_base<
          right_preconditioned_volume_operator<MatrixType, RightPreconditionerType>, Rhs,
          generic_product_impl<right_preconditioned_volume_operator<MatrixType, RightPreconditionerType>, Rhs>> {
    using Replacement = right_preconditioned_volume_operator<MatrixType, RightPreconditionerType>;
    using Scalar = typename Product<Replacement, Rhs>::Scalar;
    using Vector = EMW::Types::VectorX<Scalar>;

    template <typename Dest>
    static void scaleAndAddTo(Dest &dst, const Replacement &lhs, const Rhs &rhs, const Scalar &alpha) {
        // Right preconditioning means that the physical vector is formed first:
        // z = P C^{-1} P rhs, followed by the original volume operator P(I-KD)P.
        const Vector physical_vector = lhs.apply_right_preconditioner(rhs);
        const Vector contrast_vector = physical_vector.cwiseProduct(lhs.get_epsilon_vec());
        const Vector integral_term = lhs.get_mat() * contrast_vector;
        const Vector result = physical_vector - integral_term.cwiseProduct(lhs.get_mask());

        // Eigen may call this routine with alpha != 1 while GMRES updates its residual.
        dst.noalias() += alpha * result;
    }
};

// ------------- Simple Replacement --------------

template<typename matrix_t>
using simple_matrix_replacement = EMW::Math::LinAgl::Matrix::Wrappers::SimpleVolumeOperator<matrix_t>;

template <typename MatrixType1, typename Rhs>
struct generic_product_impl<simple_matrix_replacement<MatrixType1>, Rhs, SparseShape, DenseShape,
                            GemvProduct> // GEMV stands for matrix-vector
    : generic_product_impl_base<
          simple_matrix_replacement<MatrixType1>, Rhs,
          generic_product_impl<simple_matrix_replacement<MatrixType1>, Rhs>> {
    typedef typename Product<simple_matrix_replacement<MatrixType1>, Rhs>::Scalar Scalar;

    template <typename Dest>
    static void scaleAndAddTo(Dest &dst, const simple_matrix_replacement<MatrixType1> &lhs,
                              const Rhs &rhs, const Scalar &alpha) {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        // Для оператора объемного уравнения пишем dst += (I - K * eps) P^{-1} * rhs
#if DEBUG
        assert(alpha == Scalar(1) && "scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);
#endif
        //            VectorX<Scalar> tmp = preconditioned.cwiseProduct(lhs.get_epsilon_vec());
        dst.noalias() += rhs - (lhs.get_mat() * rhs).cwiseProduct(lhs.get_epsilon_vec());
    }
};
}

#endif //RESEARCH_MATRUXREPLACEMENT_HPP
