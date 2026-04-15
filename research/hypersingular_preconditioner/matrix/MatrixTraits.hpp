//
// Created by evgen on 07.08.2025.
//

#ifndef MATRIXTRAITS_HPP
#define MATRIXTRAITS_HPP


#include "MatrixPreconditioned.hpp"
#include "types/Types.hpp"
#include <Eigen/Core>

namespace Eigen::internal {
namespace Mat = Research::Matrix;
namespace My = Mat::Wrappers;

template <typename MatrixType, typename PrecondType> struct traits<My::MatrixReplacementReal<MatrixType, PrecondType>> {
    // Эти три вещи ниже должны зависеть от типа, который мы передаем в качестве MatrixType.
    // Но пока что это просто захардкожено
    typedef EMW::Types::scalar Scalar;
    typedef EMW::Types::integer StorageIndex;
    static constexpr EMW::Types::integer Options_ = 0;
    // --- //
    typedef Sparse StorageKind;
    typedef MatrixXpr XprKind;
    enum {
        RowsAtCompileTime = Dynamic,
        ColsAtCompileTime = Dynamic,
        MaxRowsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic,
        Flags = Options_ | NestByRefBit | LvalueBit | CompressedAccessBit,
    };
};

template <typename MatrixType, typename PrecondType> struct traits<My::MatrixReplacementComplex<MatrixType, PrecondType>> {
    // Эти три вещи ниже должны зависеть от типа, который мы передаем в качестве MatrixType.
    // Но пока что это просто захардкожено
    typedef EMW::Types::complex_d Scalar;
    typedef EMW::Types::integer StorageIndex;
    static constexpr EMW::Types::integer Options_ = 0;
    // --- //
    typedef Sparse StorageKind;
    typedef MatrixXpr XprKind;
    enum {
        RowsAtCompileTime = Dynamic,
        ColsAtCompileTime = Dynamic,
        MaxRowsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic,
        Flags = Options_ | NestByRefBit | LvalueBit | CompressedAccessBit,
    };
};

} // namespace Eigen::internal


#endif //MATRIXTRAITS_HPP
