//
// Created by evgen on 05.02.2025.
//

#include <string>

#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/matrix/iterative_solvers_coverage/DiagonalPreconditioner.hpp"
#include "math/matrix/iterative_solvers_coverage/MatrixReplacement.hpp"
#include "math/matrix/iterative_solvers_coverage/MatrixTraits.hpp"

#include "research/Solve.hpp"

#include "geometry/PeriodicStructure.hpp"

#include "VTKFunctions.hpp"

#include "GeneralEquation.hpp"
#include "FieldCalculation.hpp"
#include "FieldOverGeometry.hpp"

#include "math/matrix/Matrix.hpp"

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

template <typename Fields> void getSigmaValuesYZ(const Types::complex_d k, const Fields &fields, const std::string path) {
    int samples = 720;
    Containers::vector<Types::scalar> esas;
    esas.reserve(samples);
    Containers::vector_d angles;
    angles.reserve(samples);

    for (int i = 0; i < samples; i++) {
        Types::scalar angle = i * Math::Constants::PI<Types::scalar>() * 2 / samples;
        Types::Vector3d tau = {0, std::sin(angle), std::cos(angle)};
        esas.push_back(ESA::calculateESA(tau, k, fields.get_electric_fields(), fields.get_magnetic_fields()));
        angles.push_back(angle);
    }
    std::ofstream sigma(path + "sigmaYZ.csv");

    Utils::to_csv(esas, angles, "sigma", "angle", sigma);
}

namespace LAMatrix = Math::LinAgl::Matrix;
// каким методом расчитываем матрицу
constexpr Research::Lattice::CalculationMethod calc_method = Research::Lattice::CalculationMethod::ACA;
using TTBMatrix = Research::Lattice::CalcTraits<calc_method>::ReturnType;
// предобуславливание
using DiagonalPrec = LAMatrix::Preconditioning::DiagonalPreconditioner<Types::complex_d, TTBMatrix>;
using BlockDiagPrec = LAMatrix::Preconditioning::BlockDiagonalPreconditioner<Types::complex_d, TTBMatrix>;
using NoPrec = LAMatrix::Preconditioning::IdentityPreconditioner<Types::complex_d, TTBMatrix>;
using MatrixWrapper = LAMatrix::Wrappers::MatrixReplacement<TTBMatrix, DiagonalPrec>;

int main() {
    // считываем сетку на антенне
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/8000_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/2000_cells.csv";

    // собираем сетки
    const auto parser_out = EMW::Parser::parse_mesh_without_tag(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells};

    constexpr Types::index N1 = 9;  // количество строк в решетке
    constexpr Types::index N2 = 7;  // количество столбцов в решетке
    constexpr Types::index N1_x_N2 = N1 * N2;
    constexpr Types::scalar d1 = 0.04;  // расстояние между строками
    constexpr Types::scalar d2 = 0.08;   // расстояние между столбцами
    const Types::Vector3d dir1 = Types::Vector3d{0, -1, 0}.normalized();
    const Types::Vector3d dir2 = Types::Vector3d{1, 0, 0}.normalized();
    const Geometry::PeriodicStructure<N1, N2> geometry{dir1, dir2, d1, d2, mesh_base};

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
#if 1
    // собираем общую маленькую тёплицеву матрицу
    Research::Lattice::CalcTraits<calc_method>::error_controller = 0.001;

    auto start = std::chrono::high_resolution_clock::now();

    const auto matrix = Research::Lattice::getMatrix<calc_method>(geometry, a, k);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << Utils::get_memory_usage(matrix) << std::endl;
    std::cout << "Matrix assembled, size: " << matrix.rows() << "; time elapsed: " << elapsed << std::endl;
#endif
    // собираем правую часть шаманским способом (очень шаманским)
    // решаем какой будет фазовый фактор на волноводах
    const Containers::array<Types::scalar, N1_x_N2> phases{0};
    Containers::array<Types::complex_d, N1_x_N2> phase_factors;
    for (Types::index i = 0; i < phases.size(); ++i)
        phase_factors[i] = std::exp(Math::Constants::i * 0. * Math::Constants::deg_to_rad<Types::scalar>());
    const auto rhs = eq::getRhs(phase_factors, mesh_base, a, k);

    std::cout << "RHS assembled, size: " << rhs.rows() << "; rhs norm = " << rhs.norm() << std::endl;
#if 1
    const MatrixWrapper wrapped_mat{matrix};
    auto result = Research::solve<Eigen::GMRES>(wrapped_mat, rhs, 2000, 1e-2);

    // Разбиваем на токи и рисуем на разных многообразиях
    const Research::Lattice::FieldOver field_set(geometry, std::move(result));

    const std::string path = "/home/evgen/Education/MasterDegree/thesis/my_paper/Toeplitz_structure/numerical_results/9_7_lattice/aca/";
    const std::string dir_name = std::to_string(N1) + "_x_" + std::to_string(N2) + "_lattice/";
    VTK::set_of_fields_snapshot(field_set, path + std::to_string(N1) + "_x_" + std::to_string(N2) + "_lattice.vtu");

    getSigmaValuesXZ(k, field_set, path + dir_name);
    getSigmaValuesYZ(k, field_set, path + dir_name);
#endif
}
