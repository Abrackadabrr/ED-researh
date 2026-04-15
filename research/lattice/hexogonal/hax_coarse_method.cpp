//
// Created by evgen on 24.06.2025.
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

#include "../FieldCalculation.hpp"
#include "../FieldOverGeometry.hpp"
#include "../GeneralEquation.hpp"

#include "math/matrix/Matrix.hpp"

#include <unsupported/Eigen/IterativeSolvers>

#include "experiment/PhysicalCondition.hpp"

#include "HexagonalUtils.hpp"

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

template <typename Fields>
void getSigmaValuesXZ(const Types::complex_d k, const Fields &fields, const std::string path) {
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

template <typename Fields>
void getSigmaValuesYZ(const Types::complex_d k, const Fields &fields, const std::string path) {
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

    constexpr Types::index N1 = 12;
    constexpr Types::index N2 = 12;
    constexpr Types::index N1_x_N2 = N1 * N2;
    // а какие волноводы лишние?
    Containers::set<Types::index> out_waveguides{9, 10, 11, 22, 23, 35, 108, 120, 121, 132, 133, 134};

    const Types::scalar a_hat = 0.04; // расстояние между центрами сеток "на диагонали 1"
    const Types::scalar b_hat = 0.08; // расстояние между центрами сеток "на диагонали 2"
    const Types::scalar step = std::sqrt(a_hat * a_hat + b_hat * b_hat / 4);

    const Types::scalar alpha = std::atan2(b_hat, 2 * a_hat);
    const Types::scalar y_coord_of_vector = (2 * a_hat / b_hat);
    const Types::Vector3d dir1 = Types::Vector3d{1, y_coord_of_vector, 0}.normalized();
    const Types::Vector3d dir2 = Types::Vector3d{1, -y_coord_of_vector, 0}.normalized();

    const Scene<N1, N2> geometry{dir1, dir2, step, step, mesh_base};

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

    // собираем правую часть шаманским способом (очень шаманским)
    // решаем какой будет фазовый фактор на волноводах
    const Types::scalar phi = 0;
    auto phase_factors = Research::Lattice::Hexagonal::get_linear_phase_factors(geometry, {1, 1, 0}, phi);
    /////////////////////// и вот тут мы зануляем специально правые части на ненужных волноводах
    for (auto &&i : out_waveguides)
        phase_factors[i] = 0; // это и есть шаманское зануление
    const auto rhs = eq::getRhs(phase_factors, mesh_base, a, k);

    std::cout << "Угол отклонения диаграммы направленности по dir1: "
              << std::asin(phi * Math::Constants::deg_to_rad<Types::scalar>() / (k.real() * a_hat)) /
                     Math::Constants::deg_to_rad<Types::scalar>()
              << std::endl;
    std::cout << "Угол отклонения диаграммы направленности по dir2: "
              << std::asin(phi * Math::Constants::deg_to_rad<Types::scalar>() / (k.real() * b_hat)) /
                     Math::Constants::deg_to_rad<Types::scalar>()
              << std::endl;

    // std::copy(phase_factors.begin(), phase_factors.end(), std::ostream_iterator<Types::complex_d>(std::cout, " "));
    std::cout << "RHS assembled, size: " << rhs.rows() << std::endl;

    // собираем общую маленькую тёплицеву матрицу, притом сжатую
    Research::Lattice::CalcTraits<calc_method>::error_controller = 0.001;
    auto start = std::chrono::high_resolution_clock::now();

    const auto matrix = Research::Lattice::getMatrix<calc_method>(geometry, a, k);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << Utils::get_memory_usage(matrix) << std::endl;
    std::cout << "Matrix assembled, size: " << matrix.rows() << "; time elapsed: " << elapsed << std::endl;

    start = std::chrono::high_resolution_clock::now();

    auto result = Research::solve<Eigen::GMRES>(MatrixWrapper{matrix, out_waveguides, N1_x_N2}, rhs, 2000, 1e-2);

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << "Time elapsed for solving: " << elapsed << std::endl;

    // Разбиваем на токи и рисуем на разных многообразиях
    const Research::Lattice::FieldOver field_set(geometry, std::move(result));

    const std::string path = "/home/evgen/Education/MasterDegree/thesis/my_papers/Toeplitz_structure/numerical_results/fal/";
    const std::string dir_name = std::to_string(N1) + "_x_" + std::to_string(N2) + "_lattice_fal/";
    VTK::set_of_fields_snapshot(field_set, path + std::to_string(N1) + "_x_" + std::to_string(N2) + "_lattice_fal.vtu");

    getSigmaValuesXZ(k, field_set, path + dir_name);
    getSigmaValuesYZ(k, field_set, path + dir_name);
}
