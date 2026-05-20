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
    constexpr Types::index Nx = 2;
    constexpr Types::index Ny = 2;
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
#define FAR_ZONE 0
#if FAR_ZONE
    // Сравнение расчета взаимодействия в дальней зоне и расчета с выделением особенности
    std::cout << "// --------- Far zone interaction ------------- //" << std::endl;

    for (size_t index = 1; index < Nz - 1; ++index) {
        auto full_precise_res = operator_k.galerkin_block_for_cubes(reference_index, index);
        auto full_far_zone_result = operator_k.far_zone_interaction(reference_index, index, 3);
        std::cout << "Adaptive rel error between precise and inf = "
                  << (full_precise_res - full_far_zone_result).norm() / full_precise_res.norm() << std::endl;
    }
#endif

#define VOLUME_SINGULARITY_EXTRACTION 0
#if VOLUME_SINGULARITY_EXTRACTION
    {
        // Сравнение расчета взаимодействия в дальней зоне и расчета с выделением особенности
        std::cout << "// --------- Volume interaction ------------- //" << std::endl;
        const auto &k_corner = cubes.leftDownCorner(reference_index);
        const auto &k_center = cubes.getCells()[reference_index].center_;
        for (size_t index = 0; index < Nz - 1; ++index) {
            const auto &p_corner = cubes.leftDownCorner(index);
            auto res_se = operator_k.volume_part_singularity_extraction(k_corner, k_center, p_corner);
            auto res_naive = operator_k.volume_part_naive(k_corner, p_corner);
            std::cout << "Adaptive rel error between se and naive = " << std::abs(res_se - res_naive) / std::abs(res_se)
                      << std::endl;
        }
    }
#endif

#define SURFACE_SINGULARITY_EXTRACTION 1
#if SURFACE_SINGULARITY_EXTRACTION
    {
        // Сравнение расчета взаимодействия с и без выделения особенности
        std::cout << "// --------- Surface interaction ------------- //" << std::endl;
        const auto &cube_k = cubes.getCells()[reference_index];

        for (size_t index = 0; index < Nz - 1; ++index) {
            const auto &cube_p = cubes.getCells()[index];
            auto res_se = operator_k.surface_part_singularity_extraction(cube_k, cube_p);
        auto res_naive = operator_k.surface_part_naive(cube_k, cube_p);
        std::cout << "Adaptive rel error between se and naive = "
                  << (res_se - res_naive).norm() / res_se.norm() << std::endl;
        }
    }
#endif
}
