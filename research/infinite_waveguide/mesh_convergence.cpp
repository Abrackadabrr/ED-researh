//
// Created by evgen on 04.08.2025.
//
#include <execution>
#include <string>

#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/fields/SurfaceVectorField.hpp"
#include "math/matrix/iterative_solvers_coverage/DiagonalPreconditioner.hpp"
#include "math/matrix/iterative_solvers_coverage/MatrixReplacement.hpp"
#include "math/matrix/iterative_solvers_coverage/MatrixTraits.hpp"

#include "research/Solve.hpp"

#include "geometry/PeriodicStructure.hpp"

#include "VTKFunctions.hpp"

#include "../lattice/SpecificLatticeEquations.hpp"
#include "FieldCalculation.hpp"

#include <unsupported/Eigen/IterativeSolvers>

#include "experiment/PhysicalCondition.hpp"

#include "Equations.hpp"

#include "experiment/SWC.hpp"

#include "Utils.hpp"

#include <chrono>
#include <iostream>

#define FIELD_CALCULATION 1

template <typename Callable>
void getErrorOverAxis(const Callable &analytical_solution_f, const Math::SurfaceVectorField &j_e,
                      const Math::SurfaceVectorField &j_m, const Types::complex_d k, const std::string path) {
    // собрать точки на оси
    int N = 3000;
    Types::scalar h = 0.1428869017 * 3 / N;
    const auto points_view = std::views::iota(0, N) | std::views::transform([&h](auto p) {
                                 return Types::Vector3d{0, 0, (p + 1. / 2) * h};
                             });
    const auto z_values_view = points_view | std::views::transform([&](auto p) { return p.z(); });
    const Containers::vector<Mesh::point_t> points{std::begin(points_view), std::end(points_view)};
    const Containers::vector<Types::scalar> z_coordinate{std::begin(z_values_view), std::end(z_values_view)};

    Containers::vector<Types::scalar> amplitude{};
    std::transform(points.begin(), points.end(), std::back_inserter(amplitude), [&](auto p) {
        // эта функция, которая считает ошибку относительно аналитического решения
        Types::Vector3c calculated_value = InfiniteWaveguide::getE_in_point(j_e, j_m, k, p);
        Types::Vector3c an_value = analytical_solution_f(p);
        return (calculated_value - an_value).norm() / calculated_value.norm();
    });

    std::ofstream file(path + "/error_norm_over_z.csv");
    Utils::to_csv(amplitude, z_coordinate, "er", "z", file);
}

void getFieldNormOverAxis(const Math::SurfaceVectorField &j_e, const Math::SurfaceVectorField &j_m,
                      const Types::complex_d k, const std::string path) {
    // собрать точки на оси
    int N = 3000;
    Types::scalar h = 0.1428869017 * 3 / N;
    const auto points_view = std::views::iota(0, N) | std::views::transform([&h](auto p) {
                                 return Types::Vector3d{0, 0, (p + 1. / 2) * h};
                             });
    const auto z_values_view = points_view | std::views::transform([&](auto p) { return p.z(); });
    const Containers::vector<Mesh::point_t> points{std::begin(points_view), std::end(points_view)};
    const Containers::vector<Types::scalar> z_coordinate{std::begin(z_values_view), std::end(z_values_view)};

    Containers::vector<Types::scalar> amplitude{};
    std::transform(points.begin(), points.end(), std::back_inserter(amplitude), [&](auto p) {
        // эта функция, которая считает ошибку относительно аналитического решения
        Types::Vector3c calculated_value = InfiniteWaveguide::getE_in_point(j_e, j_m, k, p);
        return calculated_value.norm();
    });

    std::ofstream file(path + "/field_norm_over_z.csv");
    Utils::to_csv(amplitude, z_coordinate, "f_norm", "z", file);
}

void getRealFieldNormOverAxis(const Math::SurfaceVectorField &j_e, const Math::SurfaceVectorField &j_m,
                      const Types::complex_d k, const std::string path) {
    // собрать точки на оси
    int N = 3000;
    Types::scalar h = 0.1428869017 * 3. / N;
    const auto points_view = std::views::iota(0, N) | std::views::transform([&h](auto p) {
                                 return Types::Vector3d{0, 0, (p + 1. / 2) * h};
                             });
    const auto z_values_view = points_view | std::views::transform([&](auto p) { return p.z(); });
    const Containers::vector<Mesh::point_t> points{std::begin(points_view), std::end(points_view)};
    const Containers::vector<Types::scalar> z_coordinate{std::begin(z_values_view), std::end(z_values_view)};

    Containers::vector<Types::scalar> amplitude{};
    std::transform(points.begin(), points.end(), std::back_inserter(amplitude), [&](auto p) {
        // эта функция, которая считает ошибку относительно аналитического решения
        Types::Vector3c calculated_value = InfiniteWaveguide::getE_in_point(j_e, j_m, k, p);
        return calculated_value.real().y();
    });

    std::ofstream file(path + "/y_real_field_over_z.csv");
    Utils::to_csv(amplitude, z_coordinate, "f_norm", "z", file);
}

int main() {
    // считываем сетку на waveguide
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "waveguide/infinite/three_waves/5724_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "waveguide/infinite/three_waves/5722_cells.csv";
    const Types::index N_WAVES = 3;
    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells, parser_out.tags};
    mesh_base.setName("All");
    auto mesh_zero = mesh_base.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    mesh_zero.setName("Zero");

    // Геометрические параметры антенн
    // Короткая сторона волновода
    const Types::scalar a = 0.07;
    // Физика волны в пространстве
    // частота в гигагерцах
    const Types::scalar freq = Math::Constants::c / 1e8;
    const Types::complex_d k{Physics::get_k_on_frquency(freq), 0};
    // расчет коэффициента импеданса
    const Types::complex_d beta = std::sqrt(k * k - (EMW::Math::Constants::PI_square<Types::scalar>() / (a * a)));
    std::cout.precision(10);
    const Types::scalar lambda_in = 2 * Math::Constants::PI<Types::scalar>() / beta.real();
    std::cout << "Волновое число в волноводе: " << beta.real() << "; Длина волны в волноводе: " << lambda_in
              << std::endl;
    std::cout << "Волновое число в свободном пространстве: " << k.real()
              << "; Длина волны в свободном пространстве: " << 2 * Math::Constants::PI<Types::scalar>() / k.real()
              << std::endl;
    // analytical solution
    const auto get_e_h10_mode = [&k, &a, &beta](const Mesh::point_t &x) {
        const auto pi = Math::Constants::PI<Types::scalar>();
        const Types::complex_d mult = Math::Constants::i * (a / pi) * k / Math::Constants::e_0_c;
        const Types::complex_d exp = std::exp(Math::Constants::i * beta * (x.z()));
        const Types::scalar cos = std::cos(pi * x.x() / a);
        return Types::Vector3c{Types::complex_d{0, 0}, mult * cos * exp, Types::complex_d{0, 0}};
    };

    const auto rhs = InfiniteWaveguideEquations::getRhs(mesh_base, a, k);
    std::cout << "RHS assembled, size: " << rhs.rows() << std::endl;

    // собираем общую матрицу

    auto start = std::chrono::high_resolution_clock::now();

    const auto matrix = WaveGuideWithActiveSection::diagonal(mesh_base, a, k);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << "Matrix assembled, size: " << matrix.rows() << "; time elapsed: " << elapsed << std::endl;

    auto result = Research::solve<Eigen::GMRES>(matrix, rhs, 2000, 1e-3);

    // Разбиваем на токи и рисуем на разных многообразиях
    const Types::VectorXc electric_current = result.block(0, 0, 2 * mesh_base.getCells().size(), 1);
    const Types::VectorXc magnetic_current =
        result.block(2 * mesh_base.getCells().size(), 0, 2 * mesh_zero.getCells().size(), 1);

    Math::SurfaceVectorField e_c = Math::SurfaceVectorField::TangentField(mesh_base, electric_current);
    e_c.setName("e_c");
    Math::SurfaceVectorField m_c = Math::SurfaceVectorField::TangentField(mesh_zero, magnetic_current);
    m_c.setName("m_c");

    const std::string path = "/home/evgen/Education/MasterDegree/thesis/results/infinite_waveguide/wavelength_study/";

    VTK::united_snapshot<Math::SurfaceScalarField<Types::scalar>>({}, {e_c}, mesh_base, path);
    VTK::united_snapshot<Math::SurfaceScalarField<Types::scalar>>({}, {m_c}, mesh_zero, path);

    getErrorOverAxis(get_e_h10_mode, e_c, m_c, k, path);
    getFieldNormOverAxis(e_c, m_c, k, path);
    getRealFieldNormOverAxis(e_c, m_c, k, path);

#define FIELD_CALCULATION 1
#if FIELD_CALCULATION
    // Рисуем картину поля в плоскости y = 0
    int k1 = 149;
    Types::scalar h1 = lambda_in * N_WAVES / k1;
    int k2 = 40;
    Types::scalar h2 = 0.07 / k2;

    std::vector<Mesh::point_t> points;
    points.reserve(k1 * k2);
    Mesh::Utils::cartesian_product_unevenXZ(std::ranges::views::iota(0, k2), std::ranges::views::iota(0, k1),
                                            std::back_inserter(points), k2, k1, h2, h1);

    for (auto &&point : points)
        point += Mesh::point_t{0, 0, (lambda_in * N_WAVES) / 2};

    std::cout << "Surrounding Mesh Constructed" << std::endl;

    Containers::vector<Types::Vector3c> calculated_field;
    calculated_field.resize(points.size());
    Containers::vector<Types::Vector3c> analytical_field;
    analytical_field.resize(points.size());
    Containers::vector<Types::Vector3c> difference_field;
    difference_field.resize(points.size());
    Containers::vector<Types::scalar> difference_norm;
    difference_norm.resize(points.size());

#pragma omp parallel for
    for (int i = 0; i < points.size(); ++i) {
        calculated_field[i] = InfiniteWaveguide::getE_in_point(e_c, m_c, k, points[i]);
        analytical_field[i] = get_e_h10_mode(points[i]);
        difference_field[i] = calculated_field[i] - analytical_field[i];
        difference_norm[i] = difference_field[i].norm();
    }

    VTK::field_in_points_snapshot({calculated_field, analytical_field, difference_field}, {difference_norm}, {"E", "E_an", "Diff"}, {"Diff_norm"}, points, "surrounding_mesh", path);
#endif
}
