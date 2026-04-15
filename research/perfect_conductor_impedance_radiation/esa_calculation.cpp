//
// Created by evgen on 10.01.2025.
//
//
// Created by evgen on 24.12.2024.
//

#include <string>

#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/fields/SurfaceVectorField.hpp"

#include "experiment/ESA.hpp"
#include "experiment/SWC.hpp"

#include "Equations.hpp"
#include "Utils.hpp"
#include "VTKFunctions.hpp"

#include "meshes/plate/PlateGrid.hpp"

#include "operators/OperatorK.hpp"
#include "operators/OperatorR.hpp"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include "experiment/PhysicalCondition.hpp"

#include <chrono>
#include <iostream>
#include <math/integration/gauss_quadrature/GaussLegenderPoints.hpp>

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

void getSigmaValuesXZ(const Types::complex_d k, const Math::SurfaceVectorField &j_e,
                      const Math::SurfaceVectorField &j_m) {
    int samples = 360;
    Containers::vector<Types::scalar> esas;
    esas.reserve(samples);
    Containers::vector_d angles;
    angles.reserve(samples);

    for (int i = 0; i < samples; i++) {
        Types::scalar angle = (i * Math::Constants::PI<Types::scalar>() * 2 / samples) - Math::Constants::PI<Types::scalar>();
        Types::Vector3d tau = {std::sin(angle), 0, std::cos(angle)};
        esas.push_back(ESA::calculateESA(tau, k, j_e, j_m));
        angles.push_back(angle);
    }
    std::ofstream sigma("/home/evgen/Education/MasterDegree/thesis/results/perfect_conductor_impedance_radiation/sigmaXZ.csv");

    Utils::to_csv(esas, angles, "sigma", "angle", sigma);
}

void getSigmaValuesYZ(const Types::complex_d k, const Math::SurfaceVectorField &j_e,
                      const Math::SurfaceVectorField &j_m) {
    int samples = 360;
    Containers::vector<Types::scalar> esas;
    esas.reserve(samples);
    Containers::vector_d angles;
    angles.reserve(samples);

    for (int i = 0; i < samples; i++) {
        Types::scalar angle = (i * Math::Constants::PI<Types::scalar>() * 2 / samples) - Math::Constants::PI<Types::scalar>();
        Types::Vector3d tau = {0, std::sin(angle), std::cos(angle)};
        esas.push_back(ESA::calculateESA(tau, k, j_e, j_m));
        angles.push_back(angle);
    }
    std::ofstream sigma("/home/evgen/Education/MasterDegree/thesis/results/perfect_conductor_impedance_radiation/sigmaYZ.csv");

    Utils::to_csv(esas, angles, "sigma", "angle", sigma);
}

int main() {
    // считываем сетку на антенне
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "open_waveguide/18770_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "open_waveguide/18696_cells.csv";

    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile);
    auto mesh_all = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells, parser_out.tags};
    mesh_all.setName("total_mesh");
    auto mesh_zero = mesh_all.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    mesh_zero.setName("zero_mesh");

    // Короткая сторона волновода
    const Types::scalar a = 0.07;
    // Физика волны в пространстве
    // частота в гигагерцах
    const Types::scalar freq = Math::Constants::c / 1e8;
    const Types::complex_d k{Physics::get_k_on_frquency(freq), 0};
    // расчет коэффициента импеданса
    const Types::complex_d beta = std::sqrt(k * k - (EMW::Math::Constants::PI_square<Types::scalar>() / (a * a)));
    std::cout << "Волновое число в волноводе: " << beta.real()
              << "; Длина волны в волноводе: " << 2 * Math::Constants::PI<Types::scalar>() / beta.real() << std::endl;
    std::cout << "Волновое число в свободном пространстве: " << k.real()
              << "; Длина волны в свободном пространстве: " << 2 * Math::Constants::PI<Types::scalar>() / k.real()
              << std::endl;

    // тут сделать векторное поле (direct wave в волноводе)
    const auto get_e_h10_mode = [&k, &a, &beta](const Mesh::point_t &x) {
        const auto pi = Math::Constants::PI<Types::scalar>();
        const Types::complex_d mult = Math::Constants::i * (a / pi) * k / Math::Constants::e_0_c;
        const Types::complex_d exp = std::exp(Math::Constants::i * beta * (x.z()));
        const Types::scalar sin = std::cos(pi * (x.x()) / a);
        return Types::Vector3c{Types::complex_d{0, 0}, mult * sin * exp, Types::complex_d{0, 0}};
    };

    // Считаем матрицу и правую часть и решаем СЛАУ
    const auto rhs = Rupor::getRhs(mesh_all, mesh_zero, get_e_h10_mode);

    std::cout << "Rhs size: " << rhs.size() << std::endl;

    const auto matrix = Rupor::getMatrix(mesh_all, mesh_zero, a, k);

    std::cout << "Matrix assembled" << std::endl;

    const auto result = solve(matrix, rhs, 1e-2);

    // Разбиваем на два тока и рисуем на разных многообразиях
    const Types::VectorXc electric_current = result.block(0, 0, 2 * mesh_all.getCells().size(), 1);
    const Types::VectorXc magnetic_current =
        result.block(2 * mesh_all.getCells().size(), 0, 2 * mesh_zero.getCells().size(), 1);

    const std::string path = "/home/evgen/Education/MasterDegree/thesis/results/perfect_conductor_impedance_radiation/";

    Math::SurfaceVectorField e_c = Math::SurfaceVectorField::TangentField(mesh_all, electric_current);
    e_c.setName("e_c");
    Math::SurfaceVectorField m_c = Math::SurfaceVectorField::TangentField(mesh_zero, magnetic_current);
    m_c.setName("m_c");

    VTK::united_snapshot<Math::SurfaceScalarField<Types::scalar>>({}, {e_c}, mesh_all, path);
    VTK::united_snapshot<Math::SurfaceScalarField<Types::scalar>>({}, {m_c}, mesh_zero, path);

    getSigmaValuesXZ(k, e_c, m_c);
    getSigmaValuesYZ(k, e_c, m_c);
}
