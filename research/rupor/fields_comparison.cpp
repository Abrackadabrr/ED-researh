//
// Created by evgen on 10.01.2025.
//

#include <string>

#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"
#include "meshes/plate/PlateGrid.hpp"

#include "math/fields/SurfaceVectorField.hpp"

#include "Equations.hpp"
#include "FieldCalculation.hpp"

#include "experiment/PhysicalCondition.hpp"

#include "VTKFunctions.hpp"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <third_party/csv/csv.h>
#include <unsupported/Eigen/IterativeSolvers>

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

decltype(auto) getVerificationData() {
    const std::string &e_im_file =
        "/home/evgen/Education/MasterDegree/thesis/rupor_with_waveguide/comparison/space/csv/E_im.csv";
    const std::string &e_real_file =
        "/home/evgen/Education/MasterDegree/thesis/rupor_with_waveguide/comparison/space/csv/E_real.csv";
    const std::string &grid_f =
        "/home/evgen/Education/MasterDegree/thesis/rupor_with_waveguide/comparison/space/csv/grid.csv";
    io::CSVReader<3> realF(e_real_file);
    io::CSVReader<3> imF(e_im_file);
    io::CSVReader<3> gridF(grid_f);

    // я просто знаю свыше что будет 800;
    Containers::vector<Types::Vector3c> field{};
    field.reserve(800);
    Containers::vector<Types::Vector3d> points{};
    points.reserve(800);

    realF.read_header(io::ignore_extra_column, "x", "y", "z");
    imF.read_header(io::ignore_extra_column, "x", "y", "z");
    gridF.read_header(io::ignore_extra_column, "x", "y", "z");
    Types::scalar x, y, z, x1, y1, z1;

    while (realF.read_row(x, y, z)) {
        imF.read_row(x1, y1, z1);
        field.emplace_back(Types::complex_d{x, x1}, Types::complex_d{y, y1}, Types::complex_d{z, z1});
    }

    while (gridF.read_row(x, y, z)) {
        points.emplace_back(Types::Vector3d{x, y, z});
    }
    return std::pair<std::vector<Types::Vector3c>, std::vector<Types::Vector3d>>{field, points};
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
    std::cout << beta << std::endl;
    std::cout << k.real() << std::endl;

    // тут сделать векторное поле (direct wave в волноводе)
    const auto get_e_h10_mode = [&k, &a, &beta](const Mesh::point_t &x) {
        const auto pi = Math::Constants::PI<Types::scalar>();
        const Types::complex_d mult = Math::Constants::i * (a / pi) * k / Math::Constants::e_0_c;
        const Types::complex_d exp = std::exp(Math::Constants::i * beta * x.z());
        const Types::scalar sin = std::cos(pi * x.x() / a);
        return Types::Vector3c{Types::complex_d{0, 0}, mult * sin * exp, Types::complex_d{0, 0}};
    };

    // Считаем матрицу и правую часть и решаем СЛАУ
    const auto rhs = Rupor::getRhs(mesh_all, mesh_zero, get_e_h10_mode);

    const auto matrix = Rupor::getMatrix(mesh_all, mesh_sigma, mesh_zero, a, k);

    std::cout << "Matrix assembled" << std::endl;

    const auto result = solve(matrix, rhs, 1e-3);

    // Разбиваем на два тока и рисуем на разных многообразиях
    const Types::VectorXc electric_current = result.block(0, 0, 2 * mesh_all.getCells().size(), 1);
    const Types::VectorXc magnetic_current =
        result.block(2 * mesh_all.getCells().size(), 0, 2 * mesh_zero.getCells().size(), 1);

    const std::string path =
        "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/vtk_files/studies/rupor/";

    Math::SurfaceVectorField e_c = Math::SurfaceVectorField::TangentField(mesh_all, electric_current);
    e_c.setName("e_c");
    Math::SurfaceVectorField m_c = Math::SurfaceVectorField::TangentField(mesh_zero, magnetic_current);
    m_c.setName("m_c");

    VTK::united_snapshot({e_c}, {}, mesh_all, path);
    VTK::united_snapshot({m_c}, {}, mesh_zero, path);

    // Рисуем картину поля в нужных точках
    // 1) Считываем из файла нужные точки
    const auto [field, points] = getVerificationData();
    // 2) Считаем поле в этих точках
    const auto calculated_field_view = points | std::views::transform([&](auto p) {return Rupor::getE_in_point(e_c, m_c, k, p);});
    const Containers::vector<Types::Vector3c> calculated_field{calculated_field_view.begin(), calculated_field_view.end()};

    VTK::field_in_points_snapshot({field, calculated_field}, {"verification_E", "my_E"}, points, "verif_E_mesh", path);
}
