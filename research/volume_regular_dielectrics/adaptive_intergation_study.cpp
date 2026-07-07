//
// Created by evgen on 22.03.2026.
//

#include "EMW/types/Types.hpp"

#include "mesh/volume_mesh/CubeMeshWithData.hpp"

#include "operators/volume/OperatorK.hpp"

#include "experiment/PhysicalCondition.hpp"

#include "visualisation/include/VTKFunctions.hpp"

#include "MatrixReplacement.hpp"

#include "math/fields/Utils.hpp"
#include "math/integration/decart/GaussLegenderPoints.hpp"

#include "Utils.hpp"

#include "math/integration/analytical/SingularIntegration.hpp"
#include "math/integration/decart/Integration.hpp"
#include <iostream>
#include <operators/Functions.hpp>

using namespace EMW;

int main() {
    Eigen::setNbThreads(1);
    openblas_set_num_threads(1);
    // 1. Создаем одну полоску из кубов
    constexpr Types::scalar cube_length = 0.15; // размер маленького куба
    constexpr Types::index Nx = 3;
    constexpr Types::index Ny = 3;
    constexpr Types::index Nz = 100;

    Mesh::VolumeMesh::CubeMeshWithData cubes{
        Types::point_t{0, 0, 0}, (Nx - 1) * cube_length, (Ny - 1) * cube_length, (Nz - 1) * cube_length, Nx, Ny, Nz};

    // 2. Параметры падающей волны
    constexpr double lambda = 1;
    constexpr Types::complex_d k{2 * M_PI / lambda, 0.};

    const Operators::Volume::operator_K_over_cube_mesh operator_k{k, cubes};

    // 3. Адаптивный счет интегралов до заданной точности
    size_t reference_index = 0;

    // Критерии остановки
    Types::scalar rTol = 1e-5;
    Types::scalar aTol = 1e-30;

    const auto vector_stop_criterion = [rTol, aTol](const Types::Vector3c &v1, const Types::Vector3c &v2) {
        return (v1 - v2).norm() < rTol * v2.norm() + aTol;
    };
    const auto scalar_stop_criterion = [rTol, aTol](const Types::scalar &v1, const Types::scalar &v2) {
        return std::abs(v1 - v2) < rTol * std::abs(v2) + aTol;
    };
    const auto complex_stop_criterion = [](Types::scalar rTol, Types::scalar aTol) {
        return [rTol, aTol](const Types::complex_d &v1, const Types::complex_d &v2) {
            return std::abs(v1 - v2) < rTol * std::abs(v2) + aTol;
        };
    };
#define FAR_ZONE 1
#if FAR_ZONE
    // Расчет взаимодействия в дальней зоне (интеграл со внесенной производной)
    std::cout << "// --------- Far zone interaction ------------- //" << std::endl;

    for (size_t index = Nz; index < 2 * Nz - 1; ++index) {
        const auto &k_corner = cubes.leftDownCorner(reference_index);
        const auto &p_corner = cubes.leftDownCorner(index);
#if 0  // Quadrature choosing
        Types::Vector3c j = Types::Vector3c::Zero();
        j[0] = {1., 0};
        const auto integrand = [wn = k, j, cube_length](Types::scalar x1, Types::scalar y1, Types::scalar z1,
                                                        Types::scalar x2, Types::scalar y2,
                                                        Types::scalar z2) -> Types::Vector3c {
            return (cube_length * cube_length * cube_length) *
                   Helmholtz::far_zone_integral_kernel(wn, Types::point_t{x1, y1, z1} - Types::point_t{x2, y2, z2}, j);
        };
        using quad = DecartIntegration::GaussLegendre::Quadrature<2, 2, 2, 2, 2, 2>;
        auto startArgs =
            std::make_tuple(k_corner.x(), k_corner.y(), k_corner.z(), p_corner.x(), p_corner.y(), p_corner.z());
        auto deltas = std::make_tuple(cube_length, cube_length, cube_length, cube_length, cube_length, cube_length);
        auto [result, level] =
            DecartIntegration::adaptive_quadrature_sum<quad>(integrand, startArgs, deltas, vector_stop_criterion, 10);
        std::cout << (k_corner - p_corner).norm() / cube_length << ' ' << level << std::endl;
        auto result_1 = DecartIntegration::quadrature_sum_with_decomposition<quad>(integrand, startArgs, deltas, 1);
        auto result_2 = DecartIntegration::quadrature_sum_with_decomposition<quad>(integrand, startArgs, deltas, 2);
        auto result_4 = DecartIntegration::quadrature_sum_with_decomposition<quad>(integrand, startArgs, deltas, 4);
        auto result_inf = DecartIntegration::quadrature_sum_with_decomposition<quad>(integrand, startArgs, deltas, 8);
        std::cout << "Res norm = " << result_inf.norm() << std::endl;
        std::cout << "Adaptive rel error between result and 1 = " << (result_1 - result).norm() / result.norm()
                  << std::endl;
        std::cout << "Adaptive rel error between result and 2 = " << (result_2 - result).norm() / result.norm()
                  << std::endl;
        std::cout << "Adaptive rel error between result and 4 = " << (result_4 - result).norm() / result.norm()
                  << std::endl;
        std::cout << "Adaptive rel error between result and inf = " << (result_inf - result).norm() / result_inf.norm()
                  << std::endl;
#endif
        const auto volume_res = operator_k.matrix_3_coef(reference_index, index);
        Types::Matrix3c surface_res = -operator_k.matrix_2_coef(reference_index, index);
        Types::Matrix3c full_precise_res = surface_res;
        // и подправляем общую матрицу
        full_precise_res(0, 0) += volume_res;
        full_precise_res(1, 1) += volume_res;
        full_precise_res(2, 2) += volume_res;

        auto full_far_zone_result = operator_k.far_zone_interaction(reference_index, index, 3);
        std::cout << index << std::endl;
        std::cout << full_far_zone_result << std::endl;
        std::cout << full_precise_res << std::endl;
    }
#endif

#define ONE_OVER_R_CUBES 0
#if ONE_OVER_R_CUBES
    // Расчет самодействия 1/r (интеграл со внесенной производной)
    std::cout << "// --------- 1/r interaction ------------- //" << std::endl;
    for (size_t index = 0; index < Nz - 1; ++index) {
        const auto potential_of_cube_k = [corner = cubes.leftDownCorner(reference_index),
                                          cube_length](Types::scalar x, Types::scalar y, Types::scalar z) {
            Types::point_t point_to_calculate{
                x - corner.x() - cube_length / 2,
                y - corner.y() - cube_length / 2,
                z - corner.z() - cube_length / 2,
            };
            return Math::Integration::Analytical::newtonian_potential_of_parallelepiped(
                point_to_calculate, cube_length / 2, cube_length / 2, cube_length / 2);
        };

        const auto singular_part =
            DecartIntegration::adaptive_integrate<DecartIntegration::GaussLegendre::Quadrature<3, 3, 3>>(
                potential_of_cube_k,
                {cubes.leftDownCorner(index).x(), cubes.leftDownCorner(index).y(), cubes.leftDownCorner(index).z()},
                {cubes.dx(), cubes.dy(), cubes.dz()}, scalar_stop_criterion, 50);
        if (index == 0) {
            std::cout << "Value = " << Math::Integration::Analytical::self_newtonian_energy_over_cube(cube_length)
                      << std::endl;
            std::cout << "Analytical relative error = "
                      << std::abs(singular_part.first -
                                  Math::Integration::Analytical::self_newtonian_energy_over_cube(cube_length)) /
                             std::abs(Math::Integration::Analytical::self_newtonian_energy_over_cube(cube_length))
                      << std::endl;
        }
        std::cout << singular_part.second << std::endl;
    }
#endif

#define BOUNDED_PART 0
#if BOUNDED_PART
    std::cout << "// --------- unbounded part integration ------------- //" << std::endl;
    for (size_t index = 3; index < Nz - 1; ++index) {
        Types::scalar rTol = 1e-6;
        Types::scalar aTol = 1e-20;
        size_t max_level = 20;
        auto k_corner = cubes.leftDownCorner(index);
        auto p_corner = cubes.leftDownCorner(reference_index);
        const auto integrand_bounded_part = [wn = k](Types::scalar x1, Types::scalar y1, Types::scalar z1,
                                                     Types::scalar x2, Types::scalar y2, Types::scalar z2) {
            return Helmholtz::F(wn, {x1, y1, z1}, {x2, y2, z2});
        };
        const auto regular_part2 =
            DecartIntegration::adaptive_quadrature_sum<DecartIntegration::GaussLegendre::Quadrature<2, 2, 2, 2, 2, 2>>(
                integrand_bounded_part,
                {k_corner.x(), k_corner.y(), k_corner.z(), p_corner.x(), p_corner.y(), p_corner.z()},
                {cubes.dx(), cubes.dy(), cubes.dz(), cubes.dx(), cubes.dy(), cubes.dz()},
                complex_stop_criterion(rTol, aTol), max_level);
        std::cout << "2 level = " << regular_part2.second << ' '
                  << "F evals = " << std::pow(regular_part2.second, 6) * 64 << std::endl;
        const auto regular_part3 =
            DecartIntegration::adaptive_quadrature_sum<DecartIntegration::GaussLegendre::Quadrature<3, 3, 3, 3, 3, 3>>(
                integrand_bounded_part,
                {k_corner.x(), k_corner.y(), k_corner.z(), p_corner.x(), p_corner.y(), p_corner.z()},
                {cubes.dx(), cubes.dy(), cubes.dz(), cubes.dx(), cubes.dy(), cubes.dz()},
                complex_stop_criterion(rTol, aTol), max_level);
        std::cout << "3 level = " << regular_part3.second << ' '
                  << "F evals = " << std::pow(3 * regular_part3.second, 6) << std::endl;
        const auto regular_part4 =
            DecartIntegration::adaptive_quadrature_sum<DecartIntegration::GaussLegendre::Quadrature<4, 4, 4, 4, 4, 4>>(
                integrand_bounded_part,
                {k_corner.x(), k_corner.y(), k_corner.z(), p_corner.x(), p_corner.y(), p_corner.z()},
                {cubes.dx(), cubes.dy(), cubes.dz(), cubes.dx(), cubes.dy(), cubes.dz()},
                complex_stop_criterion(rTol, aTol), max_level);
        std::cout << "4 level = " << regular_part4.second << ' '
                  << "F evals = " << std::pow(4 * regular_part4.second, 6) << std::endl;

        std::cout << "rel error btw 3 and 2 = "
                  << std::abs(regular_part2.first - regular_part3.first) / std::abs(regular_part3.first) << std::endl;
        // const auto ideal_result = DecartIntegration::quadrature_sum_with_decomposition<
        //     DecartIntegration::GaussLegendre::Quadrature<3, 3, 3, 3, 3, 3>>(
        //     integrand_bounded_part,
        //     std::make_tuple(k_corner.x(), k_corner.y(), k_corner.z(), p_corner.x(), p_corner.y(), p_corner.z()),
        //     std::make_tuple(cubes.dx(), cubes.dy(), cubes.dz(), cubes.dx(), cubes.dy(), cubes.dz()), 10);
        // std::cout << "rel error btw 3 and ideal = "
        //   << std::abs(ideal_result - regular_part3.first) / std::abs(ideal_result) << std::endl;
    }

#endif
}
