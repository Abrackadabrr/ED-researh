//
// Created by evgen on 05.02.2025.
//

#include <string>

#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/fields/SurfaceVectorField.hpp"

#include "GeneralizedEquations.hpp"

#include "equations/EquationsOverGeometry.hpp"

#include "research/Solve.hpp"

#include "geometry/PeriodicStructure.hpp"

#include "VTKFunctions.hpp"

#include "FieldCalculation.hpp"
#include "FieldOverGeometry.hpp"
#include "GeneralizedEquations.hpp"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include "experiment/PhysicalCondition.hpp"

#include <Utils.hpp>
#include <chrono>
#include <experiment/ESA.hpp>
#include <iostream>

namespace eq = WaveGuideWithActiveSection;

template <int N1, int N2> using Scene = Geometry::PeriodicStructure<N1, N2>;

template <typename FieldTopology>
Containers::vector<Types::Vector3c>
calculateField(const FieldTopology &fields, const Containers::vector<Mesh::point_t> &points, Types::complex_d k) {
    Containers::vector<Types::Vector3c> field;
    field.resize(points.size());
#pragma omp parallel for num_threads(14) schedule(static)
    for (int i = 0; i < points.size(); ++i) {
        for (Types::index j = 0; j < fields.size(); ++j) {
            field[i] +=
                Lattice::getE_in_point(fields.get_electric_fields()[j], fields.get_magnetic_fields()[j], k, points[i]);
        }
    }
    return field;
}

template <typename Fields> void getSigmaValuesXZ(const Types::complex_d k, const Fields &fields, const std::string path) {
    int samples = 720;
    Containers::vector<Types::scalar> esas;
    esas.reserve(samples);
    Containers::vector_d angles;
    angles.reserve(samples);

    for (int i = 0; i < samples; i++) {
        Types::scalar angle = i * Math::Constants::PI<Types::scalar>() * 2 / samples;
        Types::Vector3d tau = {std::sin(angle), 0, std::cos(angle)};
        esas.push_back(ESA::calculateESA(tau, k, fields.get_electric_fields(), fields.get_magnetic_fields()));
        angles.push_back(angle);
    }
    std::ofstream sigma(path + "sigmaXZ.csv");

    Utils::to_csv(esas, angles, "sigma", "angle", sigma);
}

#include "math/matrix/iterative_solvers_coverage/DiagonalPreconditioner.hpp"
#include "math/matrix/iterative_solvers_coverage/MatrixReplacement.hpp"
#include "math/matrix/iterative_solvers_coverage/MatrixTraits.hpp"

namespace LAMatrix = Math::LinAgl::Matrix;
using MyMatrix = Types::MatrixXc;
using DiagonalPrec = LAMatrix::Preconditioning::DiagonalPreconditioner<Types::complex_d, MyMatrix>;
using BlockDiagPrec = LAMatrix::Preconditioning::BlockDiagonalPreconditioner<Types::complex_d, MyMatrix>;
using NoPrec = LAMatrix::Preconditioning::IdentityPreconditioner<Types::complex_d, MyMatrix>;
using MatrixWrapper = LAMatrix::Wrappers::MatrixReplacement<MyMatrix, DiagonalPrec>;

int main() {
    // считываем сетку на антенне
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/8000_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/2000_cells.csv";
    constexpr EMW::Types::index nNodes = 8000;
    constexpr EMW::Types::index nCells = 2000;

    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile, nNodes, nCells);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.first, parser_out.second};

    constexpr Types::index N1 = 1;
    constexpr Types::index N2 = 4;
    constexpr Types::index N1_x_N2 = N1 * N2;
    const Scene<N1, N2> geometry{0.14, 0.1, mesh_base};

    // Геометрические параметры антенн
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

    // собираем общую матрицу (большую и плотную!)
    const auto matrix = Equations::MatrixForStructure::compute(geometry, eq::diagonal, eq::submatrix, {a, k});

    std::cout << "Matrix assembled, size: " << matrix.rows() << std::endl;

    // собираем правую часть шаманским способом (очень шаманским)
    // решаем какой будет фазовый фактор на волноводах
    const Containers::array<Types::scalar, N1_x_N2> phases{0., 10, 20, 30};
    Containers::array<Types::complex_d, N1_x_N2> phase_factors;
    for (Types::index i = 0; i < phases.size(); ++i) {
        const Types::scalar radians = phases[i] * Math::Constants::deg_to_rad<Types::scalar>();
        std::cout << radians << std::endl;
        phase_factors[i] = std::exp(Math::Constants::i * radians);
    }
    const auto rhs = eq::getRhs(phase_factors, mesh_base, a, k);

    std::cout << "RHS assembled, size: " << rhs.rows() << std::endl;

    auto result = Research::solve<Eigen::GMRES>(MatrixWrapper{matrix}, rhs, 2000, 1e-2);

    std::cout << (matrix * result - rhs).norm() / rhs.norm() << std::endl;

    // Разбиваем на токи и рисуем на разных многообразиях
    const Research::Lattice::FieldOver field_set(geometry, std::move(result));

    const std::string path = "/home/evgen/Education/MasterDegree/thesis/results/investigation_over_rotation/full/";
    VTK::set_of_fields_snapshot(field_set, path + std::to_string(N1) + "_x_" + std::to_string(N2) + "_lattice.vtu");


# if 1  // РАСЧЕТ ЭЛЕКТРИЧЕСКИХ ПОЛЕЙ В ПЛОСКОСТИ
    // Рисуем картину поля в плоскости y = 0
    int k1 = 100;
    Types::scalar h1 = 1. / (k1 - 1);
    int k2 = 200;
    Types::scalar h2 = 2. / (k2 - 1);

    std::vector<Mesh::point_t> points;
    points.reserve(k1 * k2);
    Mesh::Utils::cartesian_product_unevenXZ(std::ranges::views::iota(0, k1), std::ranges::views::iota(0, k2),
                                            std::back_inserter(points), N1, N2, h1, h2);

    std::for_each(points.begin(), points.end(), [](Mesh::point_t& p) {p -= Mesh::point_t{0.5, 0, 0.5}; });

    std::cout << "Surrounding Mesh Constructed" << std::endl;

    const auto calculated_field = calculateField(field_set, points, k);

   VTK::field_in_points_snapshot({calculated_field}, {"E"}, points, "surrounding_mesh", path);
#endif
    getSigmaValuesXZ(k, field_set, path + std::to_string(N1) + "_x_" + std::to_string(N2) + "_lattice/");
}
