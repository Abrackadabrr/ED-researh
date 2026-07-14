//
// Created by evgen on 09.07.2026.
//

#include "mesh/volume_mesh/CubeMeshWithData.hpp"

#include "operators/volume/OperatorK.hpp"

#include "experiment/PhysicalCondition.hpp"

#include "Utils.hpp"

#include <chrono>
#include <omp.h>

using namespace EMW;

constexpr static Types::scalar SPHERE_RADUIS = 0.5;
constexpr static Types::scalar cube_length = 2 * SPHERE_RADUIS;

int main() {
    constexpr Types::index Nx = 11;
    constexpr Types::scalar freq = 0.3; // GHz
    constexpr Types::scalar rTol = 1e-5;
    constexpr Types::scalar aTol = 1e-20;
    constexpr Types::index lev_2d = 10;
    constexpr Types::index lev_3d = 7;
    constexpr Types::index lev_4d = 5;
    constexpr Types::index lev_6d = 3 ;
    constexpr Types::index nearness_trh = 2;
    // 1. Сбор сетки
    Eigen::setNbThreads(1);
    const Types::index Ny = Nx;
    const Types::index Nz = Nx;
    const Types::point_t min_corner = Types::point_t{-cube_length / 2, -cube_length / 2, -cube_length / 2};
    Mesh::VolumeMesh::CubeMeshWithData mesh{min_corner, cube_length, cube_length, cube_length, Nx, Ny, Nz};
    const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();
    auto basis_fn_module = 1. / sqrt(cube_measure);

    // 2. Параметры падающей волны
    constexpr Types::complex_d k{Physics::get_k_on_frquency(freq), 0.0};

    // 3. Галеркинская проекция оператора
    Operators::Volume::operator_K_over_cube_mesh operator_k{k, mesh};
    operator_k.set_nearness_threshold(nearness_trh);
    operator_k.set_tolerances(rTol, aTol);
    operator_k.set_adaptive_integration_max_levels({lev_2d, lev_3d, lev_4d, lev_6d});

    std::cout << "Maх threads available: " << omp_get_max_threads() << std::endl;

    for (int n = 1; n < 17; n++) {
        omp_set_num_threads(n);
        // auto warming_result = operator_k.compute_galerkin_matrix(basis_fn_module);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = operator_k.compute_galerkin_matrix(basis_fn_module);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "# " << n << ": " << elapsed.count() << " milliseconds" << std::endl;
        }
}
