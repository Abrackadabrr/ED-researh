//
// Created by evgen on 25.03.2026.
//

#ifndef RESEARCH_MATRIXTRAITS_HPP
#define RESEARCH_MATRIXTRAITS_HPP

#include "MatrixReplacement.hpp"

namespace Eigen::internal
{
    namespace Mat = EMW::Math::LinAgl::Matrix;
    namespace My = Mat::Wrappers;
    /**
     * Характеритики типа MatrixReplacement<MatrixType>
     * Это необходимо определенить для того, чтобы Eigen верно разрешал типы
     * @tparam MatrixType
     */
    template <typename MatrixType1, typename PrecondType>
    struct traits<My::VolumeOperatorMatrixReplacement<MatrixType1, PrecondType>>
    {
        // Эти три вещи ниже должны зависеть от типа, который мы передаем в качестве MatrixType.
        // Но пока что это просто захардкожено
        typedef EMW::Types::complex_d Scalar;
        typedef EMW::Types::integer StorageIndex;
        static constexpr EMW::Types::integer Options_ = 0;
        // --- //

        typedef Sparse StorageKind;
        typedef MatrixXpr XprKind;

        enum
        {
            RowsAtCompileTime = Dynamic,
            ColsAtCompileTime = Dynamic,
            MaxRowsAtCompileTime = Dynamic,
            MaxColsAtCompileTime = Dynamic,
            Flags = Options_ | NestByRefBit | LvalueBit | CompressedAccessBit,
        };
    };

    template <typename MatrixType1>
    struct traits<My::VolumeOperatorMatrixReplacement<MatrixType1>>
    {
        // Эти три вещи ниже должны зависеть от типа, который мы передаем в качестве MatrixType.
        // Но пока что это просто захардкожено
        typedef EMW::Types::complex_d Scalar;
        typedef EMW::Types::integer StorageIndex;
        static constexpr EMW::Types::integer Options_ = 0;
        // --- //

        typedef Sparse StorageKind;
        typedef MatrixXpr XprKind;

        enum
        {
            RowsAtCompileTime = Dynamic,
            ColsAtCompileTime = Dynamic,
            MaxRowsAtCompileTime = Dynamic,
            MaxColsAtCompileTime = Dynamic,
            Flags = Options_ | NestByRefBit | LvalueBit | CompressedAccessBit,
        };
    };
} // namespace Eigen::internal

#endif //MATRIXTRAITS_HPP
