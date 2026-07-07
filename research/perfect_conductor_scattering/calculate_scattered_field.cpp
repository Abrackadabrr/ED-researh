//
// Created by evgen on 11.06.2025.
//

#include "experiment/PhysicalCondition.hpp"
#include "math/MathConstants.hpp"
#include "math/integration/decart/Integration.hpp"
#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"
#include "mesh/Utils.hpp"
#include "slae_generation/MatrixGeneration.hpp"
#include "types/Types.hpp"
#include "visualisation/include/VTKFunctions.hpp"

#include "operators/OperatorK.hpp"
#include "slae_generation/MatrixGeneration.hpp"

#include "../Solve.hpp"

#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <Utils.hpp>
#include <experiment/ESA.hpp>
#include <fstream>
#include <iostream>
#include <ranges>
#include <unsupported/Eigen/IterativeSolvers>

using namespace EMW;
using namespace EMW::Types;

inline Types::Vector3c operatorK_in_point(const Math::SurfaceVectorField &field, const Types::complex_d k,
                                          const Mesh::point_t &point) {
    return EMW::OperatorK::K1<DecartIntegration::GaussLegendre::Quadrature<4, 4>>(point, k, field) +
           EMW::OperatorK::K0<DecartIntegration::GaussLegendre::Quadrature<4>>(point, k, field);
}

inline Types::Vector3c getE_in_point(const Math::SurfaceVectorField &j_e, const Types::complex_d k,
                                     const Mesh::point_t &point) {
    const Types::complex_d mul = Math::Constants::i / k;
    const auto value = operatorK_in_point(j_e, k, point);
    // std::cout << value << std::endl;
    return value;
}

/*

 */

int main() {
    // читаем четку
    const std::string path_to_meshes = "/home/evgen/Education/MasterDegree/thesis/ED-researh/meshes/sphere_antenna_tri/";
    const std::string nodesFile = path_to_meshes + "1213_nodes.csv";
    const std::string cellsFile = path_to_meshes + "2336_cells.csv";

    const auto parser_out = EMW::Parser::parse_mesh_without_tag(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells};
    mesh_base.setName("test_antenna");

    // физика
    const Types::scalar freq = 0.3;
    const Types::complex_d k{EMW::Physics::get_k_on_frquency(freq), 0};
    std::cout << 2 * M_PI / k.real() << std::endl;

    EMW::Physics::planeWaveCase physics(Vector3d{0, 1, 0}.normalized(),  // polarization
                                        k,                               // wave figure
                                        Vector3d{1, 0, 0}.normalized()); // wave unit vector

    // Диполь Герца
    Types::Vector3c dipole_moment{
        Types::complex_d{0, 0},
        Types::complex_d{0, 0},
        Types::complex_d{1, 0},
    };
    EMW::Physics::HertzElectricDipole dipole{dipole_moment, k, Types::Vector3d{-0.5, 0, 0}};

    // смотрим след поля на поверхности
    const auto E_0_field =
        Math::SurfaceVectorField{mesh_base, [dipole](const Types::Vector3d &point) { return dipole(point); }};
    const Types::VectorXc b = -1 * E_0_field.asVector();

    // расчет
    const MatrixXc A = EMW::Matrix::getMatrixK(k, mesh_base);
    const auto j_vec = Research::solve<Eigen::GMRES>(A, b, 2000, 1e-7);

    // закидываем ток как поле на поверхности
    const Math::SurfaceVectorField j_e = Math::SurfaceVectorField::TangentField(mesh_base, j_vec, "j_e");

    // расчитываем поле вокруг заданной геометрии

    const std::string path = "/home/evgen/Work/INM_RAS/Fidesys/final_verification/antenna/";
#define FIELD_CALCULATION 1
#if FIELD_CALCULATION
    // Смотрим на поле диполя
    Mesh::VolumeMesh::CubeMesh cube_mesh{Types::Vector3d{-2, -2, -2}, 4., 40};

    auto points = cube_mesh.getNodes();

    Containers::vector<Types::Vector3c> calculated_field{};
    calculated_field.resize(points.size());

    std::cout << "initialized" << std::endl;

#pragma omp parallel for num_threads(14) schedule(dynamic)
    for (int i = 0; i < points.size(); ++i) {
        calculated_field[i] = dipole(points[i]);
    }

    VTK::field_in_points_snapshot({calculated_field}, {}, {"E_dipole"}, {}, points, "surrounding_mesh", path);
#endif

#define BISTATIC_RCS 0
#if BISTATIC_RCS
    int N = 360;
    const auto get_tau_vv = [](Types::scalar phi) { return Types::Vector3d{std::cos(phi), 0, std::sin(phi)}; };
    const auto get_tau_hh = [](Types::scalar phi) { return Types::Vector3d{std::cos(phi), std::sin(phi), 0}; };

    auto view =
        std::views::iota(0, N) | std::views::transform([N](int i) { return i * Math::Constants::PI<double>() / N; });
    std::vector<Types::scalar> phis{view.begin(), view.end()};

    std::vector<Types::scalar> rsp_hh{};
    rsp_hh.resize(N);
    std::vector<Types::scalar> rsp_vv{};
    rsp_vv.resize(N);

#pragma omp parallel for num_threads(14)
    for (auto &&[vvalue, hvalue, phi] : std::views::zip(rsp_vv, rsp_hh, phis)) {
        vvalue = 10 * std::log(10 * sigma_check::calculateRСS(get_tau_vv(phi), k, j_e));
        hvalue = 10 * std::log(10 * sigma_check::calculateRСS(get_tau_hh(phi), k, j_e));
    }

    auto degree_view = view | std::views::transform([&](Types::scalar phi) { return phi * 180 / M_PI; });
    std::vector<Types::scalar> phis_degree{degree_view.begin(), degree_view.end()};
    std::ofstream rsp_vv_file{path + "sigma_vv.csv"};
    std::ofstream rsp_hh_file{path + "sigma_hh.csv"};
    Utils::to_csv(phis_degree, rsp_vv, "angle", "rcs", rsp_vv_file);
    Utils::to_csv(phis_degree, rsp_hh, "angle", "rcs", rsp_hh_file);
#endif

    // рисуем поле токов
    VTK::united_snapshot<Math::SurfaceScalarField<Types::complex_d>>({}, {j_e}, mesh_base, path);

}
