//
// Created by evgen on 24.12.2024.
//

#ifndef EQUATIONS_HPP
#define EQUATIONS_HPP

#include "mesh/SurfaceMesh.hpp"

#include "slae_generation/MatrixGeneration.hpp"
#include "slae_generation/MatrixInplaceGeneration.hpp"


#include "types/Types.hpp"

#include "math/fields/SurfaceVectorField.hpp"

#include <math/MathConstants.hpp>

#include <iostream>

namespace EMW::Rupor {

inline void checkNan(const Types::MatrixXc &A, const std::string &msg) {
    const Types::complex_d sum = A.sum();
    if (std::isnan(sum.real()) || std::isnan(sum.imag())) {
        std::cout << "nan, " << msg << std::endl;
    }
}

/**
 * Вычисляем матрицу для системы уравнений расчета рупорной антенны
 */
inline EMW::Types::MatrixXc getMatrix(const Mesh::SurfaceMesh &mesh_all, const Mesh::SurfaceMesh &mesh_zero,
                                      Types::scalar a, Types::complex_d k) {
    // описываем размеры итоговой матрицы
    const Types::index N = mesh_all.getCells().size();
    const Types::index K = mesh_zero.getCells().size();
    const Types::index size = 2 * (N + K);
    Types::MatrixXc A(size, size);

    const Types::complex_d epsilon{1., 0};
    const Types::complex_d mu{1., 0};

    // расчет коэффициента импеданса и вспомогательных констант
    const Types::complex_d beta = std::sqrt(k * k - EMW::Math::Constants::PI_square<Types::scalar>() / (a * a));
#ifdef OLD
    const Types::complex_d z = -k * Math::Constants::one_div_c / beta;
    const Types::complex_d with_e = Math::Constants::i * Math::Constants::mu_0_c / k;
    const Types::complex_d with_mu = Math::Constants::i * Math::Constants::e_0_c / k;
#else
    const Types::complex_d z = -k * mu / beta;
    const Types::complex_d with_e = Math::Constants::i / (k * epsilon);
    const Types::complex_d with_mu = Math::Constants::i / (k * mu);
#endif

    // генерируем необходимые подматрицы
    auto K_all_block = A.block(0, 0, 2 * N, 2 * N);
    auto R_0_block = A.block(0, 2 * N, 2 * N, 2 * K);
    auto R_all_block = A.block(2 * N, 0, 2 * K, 2 * N);
    auto K_0_block = A.block(2 * N, 2 * N, 2 * K, 2 * K);

    Matrix::getMatrixK_inplace(K_all_block, k, mesh_all);
    Matrix::getMatrixR_inplace(R_0_block, k, mesh_zero, mesh_all);
    Matrix::getMatrixK_inplace(K_0_block, k, mesh_zero);
    Matrix::getMatrixR_inplace(R_all_block, k, mesh_all, mesh_zero);

    // тривиальные операторы нужных размеров из уравнений
    // оператор векторного умножения на нормаль справа
    const Types::MatrixXc C = 0.5 * Matrix::getMatrixCrossNormal(k, mesh_zero);
    // единичный оператор
    const auto I_m = Matrix::getMatrixIdentity(k, mesh_zero);

    // 1) Дописываем слагаемые в уравнение на поверхности sigma_0
    // ВНИМАНИЕ: тут идёт существенный упор на то, что мы знаем, что sigma_0 лежит последняя в sigma_all
    R_0_block.block(N - K, 0, K, 2 * K) -= C.block(0, 0, K, 2 * K);
    R_0_block.block(2 * N - K, 0, K, 2 * K) -= C.block(K, 0, K, 2 * K);

    R_all_block.block(0, N - K, 2 * K, K) += C.block(0, 0, 2 * K, K);
    R_all_block.block(0, 2 * N - K, 2 * K, K) += C.block(0, K, 2 * K, K);

    std::cout << "mem control point" << std::endl;

    // 2) Собираем матрицу
    K_all_block *= with_e;
    R_0_block *= -Types::complex_d{1, 0};
    R_all_block *= z;
    K_0_block *= z * with_mu;
    K_0_block += I_m;

    // 3) Готово
    return A;
}

template <typename Callable>
Types::VectorXc getRhs(const Mesh::SurfaceMesh &mesh_all, const Mesh::SurfaceMesh &mesh_zero, const Callable &rhs) {
    const Types::index N = mesh_all.getCells().size();
    const Types::index K = mesh_zero.getCells().size();

    const Math::SurfaceVectorField direct_e_field{mesh_zero, rhs};
    const Types::VectorXc non_zer0_rhs = -2 * direct_e_field.normalCrossField().asVector();

    Types::VectorXc res = Types::VectorXc::Zero(2 * N + 2 * K);
    res.block(2 * N, 0, 2 * K, 1) = non_zer0_rhs;
    return res;
}
}

#endif // EQUATIONS_HPP
