//
// Created by evgen on 16.01.2025.
//

#ifndef LATTICE_MANUAL_EQUATIONS_HPP
#define LATTICE_MANUAL_EQUATIONS_HPP

#include "mesh/SurfaceMesh.hpp"

#include "slae_generation/MatrixGeneration.hpp"

#include "types/Types.hpp"

#include <math/MathConstants.hpp>

#include "research/rupor/Equations.hpp"

#include <iostream>

namespace Lattice {

using namespace EMW;

inline void checkNan(const Types::MatrixXc &A, const std::string &msg) {
    const Types::complex_d sum = A.sum();
    if (std::isnan(sum.real()) || std::isnan(sum.imag())) {
        std::cout << "nan, " << msg << std::endl;
    }
}

/**
 * Вспомогательная функция для вычисления взаимодействия двух антенн
 * Сетки mesh_to_integrate и mesh_with_collocation_points не обязаны совпадать,
 * функция будет работать для произвольных сеток
 *
 * ПОКА ЧТО НАПИШЕМ КАК БУДТО СЕТКИ ОДИНАКОВЫЕ В ТОПОЛОГИЧЕСКОМ СМЫСЛЕ (ПРИ ЭТОМ РАЗНЫЕ В ПРОСТРАНСТВЕ)
 */
Types::MatrixXc submatrix(const Mesh::SurfaceMesh &mest_to_integrate,
                          const Mesh::SurfaceMesh &mest_with_collocation_points, const Types::scalar a,
                          const Types::complex_d k) {
    // описываем специфические части сеток
    // части с электрическими токами
    const Mesh::SurfaceMesh mesh_to_integrate_all = mest_to_integrate;
    const Mesh::SurfaceMesh mesh_to_integrate_zero =
        mest_to_integrate.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    const Mesh::SurfaceMesh mesh_collocation_all = mest_with_collocation_points;
    const Mesh::SurfaceMesh mesh_collocation_zero =
        mest_with_collocation_points.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);

    // Надо понять какие размеры будем иметь итоговая подматрица
    // Сейчас предполагаем, что сетки одинаковые, поэтому размеры остаются как и были раньше
    const Types::index N = mesh_to_integrate_all.getCells().size();
    const Types::index K = mesh_to_integrate_zero.getCells().size();
    const Types::index size = 2 * (N + K);
    Types::MatrixXc A(size, size);

    const Types::complex_d epsilon{1., 0};
    const Types::complex_d mu{1., 0};

    // расчет коэффициента импеданса и вспоногательных констант
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

    // 1) генерируем необходимые подматрицы
    // а) Матрица для оператора К: теперь значения оператора К лежат на другой сетке,
    // значит и считать надо через более общую функцию
    auto K_all = Matrix::getMatrixK(k, mesh_to_integrate_all, mesh_collocation_all);
    checkNan(K_all, "K_all");
    auto R_0 = Matrix::getMatrixR(k, mesh_to_integrate_zero, mesh_collocation_all);
    checkNan(R_0, "R0");
    auto K_0 = Matrix::getMatrixK(k, mesh_to_integrate_zero, mesh_collocation_zero);
    checkNan(K_0, "K0");
    auto R_all = Matrix::getMatrixR(k, mesh_to_integrate_all, mesh_collocation_zero);
    checkNan(R_all, "R_all");

    // неаддитивные по поверхности операторы отваливаются при расчете недиагональных матриц
    // поэтому их тут не будет

    // Далее все идентично как и в случае одной антенны, блоки распихиваем точно так же
    // Если будут изменяться размеры, то придется и этот кусок ниже переписывать

    // 2) Собираем матрицу
    A.block(0, 0, 2 * N, 2 * N) = with_e * K_all;
    A.block(0, 2 * N, 2 * N, 2 * K) = -R_0;
    A.block(2 * N, 0, 2 * K, 2 * N) = z * R_all;
    A.block(2 * N, 2 * N, 2 * K, 2 * K) = z * with_mu * K_0;

    // 3) Готово
    return A;
}

/**
 * Вычисляем матрицу для системы уравнений двух одинаковых взаимодействующих антенн
 */
inline EMW::Types::MatrixXc getSLAE(const Mesh::SurfaceMesh &mesh_1, const Mesh::SurfaceMesh &mesh_2, Types::scalar a,
                                    Types::complex_d k) {
    // Матрица будет состоять из четырёх блоков
    // | A_11 | A_12 |
    // |______|______|
    // | A_21 | A_22 |

    // Уравнения формально остаются теми же, просто сначала мы опишем все уравнения, соответствующие точкам
    // коллокации для mesh_1, а потом уже для mesh_2

    // Тут нам и понадобится возможность аппроксимировать операторы в случаях, когда сетка с интегрированием отличается
    // от сетки с точками коллокации.

    // 1) Для начала рассчитаем матрицы взаимодействия "сам на себя" для обеих антенн

    const auto A11 = Rupor::getMatrix(mesh_1, mesh_1.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION), a, k);
    const auto A22 = Rupor::getMatrix(mesh_2, mesh_2.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION), a, k);

    // 2) Теперь сделаем расчет для точек коллокации на сетке mesh_1, но при этом влияние учитывается от токов на mesh_2
    // и наоборот
    const auto A12 = submatrix(mesh_2, mesh_1, a, k);
    const auto A21 = submatrix(mesh_1, mesh_2, a, k);
    const Types::index N = A11.rows();
    const Types::index size = 2 * N;
    Types::MatrixXc A(size, size);
    A.block(0, 0, N, N) = A11;
    A.block(N, N, N, N) = A22;
    A.block(N, 0, N, N) = A21;
    A.block(0, N, N, N) = A12;

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

#endif
