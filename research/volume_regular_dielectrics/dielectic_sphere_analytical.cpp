//
// Created by evgen on 22.03.2026.
//

#include "EMW/types/Types.hpp"

#include "mesh/Utils.hpp"
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
#include "math/fourier/TripleToeplitz3x3Fourier.hpp"
#include "math/integration/decart/Integration.hpp"

#include "Utils.hpp"

#include <iostream>

using namespace EMW;

constexpr Types::scalar SPHERE_RADUIS = 0.5;
constexpr Types::scalar CUBE_LENGTH = 0.5;

Types::scalar homo_sphere(const Types::point_t &x) { return x.norm() < SPHERE_RADUIS ? 2.56 : 1; }

#define CALC_RSP 1
#define CALC_FIELD 0

int main() {
    Eigen::setNbThreads(1);
    openblas_set_num_threads(1);
    // 1. Читаем сетку
    const std::string path =
        "/home/evgen/Education/MasterDegree/thesis/ED-researh/research/volume_regular_dielectrics/";
    const std::string filename = "sphere_61_with_an_sol.vtu";
    auto mesh = VTK::volume_mesh_withdata_from_vtu(path + filename);
    auto solution = mesh.getVectorData("solution");
    for (auto&& val : solution) {
        val = Math::Constants::i * val * 1.56;
    }
    mesh.setVectorData("solution_j", solution);

    // 2. Параметры падающей волны
    constexpr Types::scalar freq = 1; // GHz
    constexpr Types::complex_d k{Physics::get_k_on_frquency(freq), 0.};

    Physics::planeWaveCase incident_field{Types::Vector3d{0, 1, 0}, k, Types::Vector3d{1, 0, 0}};
    std::cout << "Длина волны в свободном пространстве = " << 2 * M_PI / k.real() << std::endl;
    std::cout << "lambda_0 / mesh.h = " << 2 * M_PI / (k.real() * mesh.dx()) << std::endl;
    std::cout << "lambda / mesh.h = " << 2 * M_PI / (2.56 * k.real() * mesh.dx()) << std::endl;


#if CALC_RSP

    for (auto&& solution_filed_name : std::vector<std::string>{"an_sol", "solution_j"}) {
        // 8. Расчситываем диаграмму направленности
        int N = 180;
        const auto get_tau_hh = [](Types::scalar phi) { return Types::Vector3d{std::cos(phi), 0, std::sin(phi)}; };
        const auto get_tau_vv = [](Types::scalar phi) { return Types::Vector3d{std::cos(phi), std::sin(phi), 0}; };

        auto view = std::views::iota(0, N) | std::views::transform([N](int i) { return i * M_PI / N; });
        std::vector<Types::scalar> phis{view.begin(), view.end()};

        std::vector<Types::scalar> rsp_hh{};
        rsp_hh.resize(N);
        std::vector<Types::scalar> rsp_vv{};
        rsp_vv.resize(N);

#pragma omp parallel for num_threads(14)
        for (auto &&[vvalue, hvalue, phi] : std::views::zip(rsp_vv, rsp_hh, phis)) {
            auto pred_value = ESA::calculateRSP(get_tau_vv(phi), k, solution_filed_name, mesh);
            vvalue = (std::log(pred_value));
            hvalue = (std::log(ESA::calculateRSP(get_tau_hh(phi), k, solution_filed_name, mesh)));
        }

        auto degree_view = view | std::views::transform([&](Types::scalar phi) { return phi * 180 / M_PI; });
        std::vector<Types::scalar> phis_degree{degree_view.begin(), degree_view.end()};
        std::ofstream rsp_vv_file{path + "sigma_vv_" + solution_filed_name + ".csv"};
        std::ofstream rsp_hh_file{path + "sigma_hh_" + solution_filed_name + ".csv"};
        Utils::to_csv(phis_degree, rsp_vv, "angle", "rsp", rsp_vv_file);
        Utils::to_csv(phis_degree, rsp_hh, "angle", "rsp", rsp_hh_file);
#endif
    }

#if CALC_FIELD
    // 9. Считаем поле
    const int N = 100;
    const Types::scalar h = 0.04;
    std::vector<Mesh::point_t> meshgrid;
    meshgrid.reserve(N * N);
    Mesh::Utils::cartesian_product_unevenXY(std::ranges::views::iota(0, N), std::ranges::views::iota(0, N),
                                            std::back_inserter(meshgrid), N, N, h, h);
    std::vector<Types::Vector3c> field;
    field.resize(meshgrid.size());

#pragma omp parallel for num_threads(14)
    for (size_t idx = 0; idx < meshgrid.size(); ++idx) {
        field[idx] = operator_K.compute_arbitrary_point(meshgrid[idx], mesh.getVectorData("solution"));
    }

    VTK::field_in_points_snapshot({field}, {}, {"E"}, {}, meshgrid, "electric_filed_" + std::to_string(Nx) , path);
#endif
};
