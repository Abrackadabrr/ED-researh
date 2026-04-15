//
// Created by evgen on 13.01.2025.
//
#include <string>

#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/fields/SurfaceVectorField.hpp"

#include "Equations.hpp"
#include "VTKFunctions.hpp"

#include "meshes/plate/PlateGrid.hpp"

#include "FieldCalculation.hpp"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include "experiment/PhysicalCondition.hpp"

#include <chrono>
#include <iostream>

using namespace EMW;

Types::VectorXc solve(const Types::MatrixXc &A, const Types::VectorXc &b, Types::scalar tolerance) {
    auto method = Eigen::GMRES<Types::MatrixXc>{};

    Types::index max_iterations = 500;

    method.setMaxIterations(max_iterations);
    std::cout << method.maxIterations() << std::endl;
    method.setTolerance(tolerance);
    method.set_restart(max_iterations);

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

    method.compute(A);
    auto j_vec = Types::VectorXc{method.solve(b)};

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::ratio<1, 1>> elapsed_seconds = end - start;
    std::cout << "Время решения GMRES: " << elapsed_seconds.count() << std::endl;
    std::cout << "total iterations: " << method.iterations() << std::endl;
    std::cout << "tolerance: " << method.tolerance() << std::endl;
    std::cout << "total error: " << method.error() << std::endl;
    std::cout << "Info: " << static_cast<int>(method.info()) << std::endl;
    return j_vec;
}

int main() {
    // считываем сетку на антенне
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "rupor/15200_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "rupor/3800_cells.csv";
    const EMW::Types::index nNodes = 15200;
    const EMW::Types::index nCells = 3800;

    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile, nNodes, nCells);
    auto mesh_all = Mesh::SurfaceMesh{parser_out.first, parser_out.second};
    mesh_all.setName("total_mesh");
    auto mesh_sigma = mesh_all.getSubmesh(Mesh::IndexedCell::Tag::SIGMA);
    mesh_sigma.setName("sigma_mesh");
    auto mesh_zero = mesh_all.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    mesh_zero.setName("zero_mesh");

    // Короткая сторона волновода
    const Types::scalar a = 0.07;
    // Физика волны в пространстве
    // частота в гигагерцах
    const Types::scalar freq = 3;
    const Types::complex_d k{Physics::get_k_on_frquency(freq), 0};
    // расчет коэффициента импеданса
    const Types::complex_d beta = std::sqrt(k * k - (EMW::Math::Constants::PI_square<Types::scalar>() / (a * a)));
    std::cout << "Волновое число в волноводе: " << beta.real() << "; Длина волны в волноводе: " << 2 * Math::Constants::PI<Types::scalar>() / beta.real() << std::endl;
    std::cout << "Волновое число в свободном пространстве: " << k.real() << "; Длина волны в свободном пространстве: " << 2 * Math::Constants::PI<Types::scalar>() / k.real() << std::endl;

    Math::SurfaceVectorField e_c =
        Math::SurfaceVectorField(mesh_all, [](const Mesh::IndexedCell &cell) -> Types::Vector3c {
            return cell.tau[0] + Math::Constants::i * Types::Vector3d::Zero();
        });
    e_c.setName("e_c");
    Math::SurfaceVectorField m_c = Math::SurfaceVectorField(mesh_zero, [](const Mesh::IndexedCell &cell) -> Types::Vector3c {
        return cell.tau[0] + Math::Constants::i * Types::Vector3d::Zero();
        });
    m_c.setName("m_c");

    const std::string path =
    "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/vtk_files/studies/rupor/";

    VTK::united_snapshot({e_c}, {}, mesh_all, path);
    VTK::united_snapshot({m_c}, {}, mesh_zero, path);

    // Рисуем картину поля в плоскости y = 0
    int N1 = 100;
    Types::scalar h1 = 1. / (N1 - 1);
    int N2 = 200;
    Types::scalar h2 = 1. / (N2 - 1);

    std::vector<Mesh::point_t> points;
    points.reserve(N1 * N2);
    Mesh::Utils::cartesian_product_unevenYZ(std::ranges::views::iota(0, N1), std::ranges::views::iota(0, N2),
                                            std::back_inserter(points), N1, N2, h1, h2);

    const auto calculated_field_view =
        points | std::views::transform([&](auto p) {
        return Rupor::getE_in_point(e_c, m_c, k, p); });
    const Containers::vector<Types::Vector3c> calculated_field{calculated_field_view.begin(),
                                                               calculated_field_view.end()};

    VTK::field_in_points_snapshot({calculated_field}, {"E",}, points, "surrounding_mesh", path);
}
