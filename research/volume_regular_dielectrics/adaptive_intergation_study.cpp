//
// Created by evgen on 22.03.2026.
//

#include "EMW/types/Types.hpp"

#include "mesh/volume_mesh/CubeMeshWithData.hpp"

#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "experiment/ESA.hpp"
#include "experiment/PhysicalCondition.hpp"

#include "../Solve.hpp"

#include "visualisation/include/VTKFunctions.hpp"

#include "MatrixReplacement.hpp"
#include "MatrixTraits.hpp"
#include "Preconditioning.hpp"

#include "math/fields/Utils.hpp"
#include "math/integration/decart/GaussLegenderPoints.hpp"

#include "Utils.hpp"

#include <iostream>
#include <math/integration/decart/Integration.hpp>
#include <operators/Functions.hpp>

using namespace EMW;

constexpr Types::scalar SPHERE_RADUIS = 0.5;
constexpr Types::scalar CUBE_LENGTH = 0.5;

Types::scalar homo_sphere(const Types::point_t &x) { return x.norm() < SPHERE_RADUIS ? 2.56 : 1; }

Types::scalar permittivity_distribution_cube(const Types::point_t &x) {
    return 3 * ((std::abs(x.x()) < CUBE_LENGTH / 2) && (std::abs(x.y()) < CUBE_LENGTH / 2) &&
                (std::abs(x.z()) < CUBE_LENGTH / 2)) +
           1;
}

int main() {
    Eigen::setNbThreads(1);
    openblas_set_num_threads(1);
    // 1. Создаем одну полоску из кубов
    constexpr Types::scalar cube_length = 0.1;
    constexpr Types::index Nx = 2;
    constexpr Types::index Ny = 2;
    constexpr Types::index Nz = 100;

    Mesh::VolumeMesh::CubeMeshWithData cubes{
        Types::point_t{0, 0, 0}, (Nx - 1) * cube_length, (Ny - 1) * cube_length, (Nz - 1) * cube_length, Nx, Ny, Nz};

    // 2. Параметры падающей волны
    constexpr double lambda = 1;
    constexpr Types::complex_d k{2 * M_PI / lambda, 0.};

    // 3. Адаптивный счет интегралов до заданной точности
    size_t reference_index = 0;

    // Критерии остановки
    Types::scalar rTol = 1e-5;
    Types::scalar aTol = 1e-16;
    const auto vector_stop_criterion = [rTol, aTol](const Types::Vector3c &v1, const Types::Vector3c &v2) {
        return (v1 - v2).norm() < rTol * v2.norm() + aTol;
    };

    // Расчет взаимодействия в дальней зоне (интеграл со внесенной производной)
    std::cout << "// --------- Far zone interaction ------------- //" << std::endl;

    for (size_t index = 1; index < Nz - 1; ++index) {
        const auto &k_corner = cubes.leftDownCorner(reference_index);
        const auto &p_corner = cubes.leftDownCorner(index);
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
        std::cout << "Adaptive rel error between result and 4 = " << (result_4 - result).norm() / result.norm() << std::endl;
        std::cout << "Adaptive rel error between result and inf = " << (result_inf - result).norm() / result_inf.norm() << std::endl;
    }
}
