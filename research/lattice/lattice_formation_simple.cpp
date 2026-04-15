//
// Created by evgen on 16.01.2025.
//

#include <string>

#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/fields/SurfaceVectorField.hpp"

#include "experiment/ESA.hpp"

#include "deprecated/Equations.hpp"

#include "VTKFunctions.hpp"

#include "meshes/plate/PlateGrid.hpp"

#include "FieldCalculation.hpp"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include "experiment/PhysicalCondition.hpp"

#include "research/Solve.hpp"

#include "Utils.hpp"

#include <iostream>

using namespace EMW;

Containers::vector<Types::Vector3c>
calculateField(const Math::SurfaceVectorField &e_c_1, const Math::SurfaceVectorField &e_c_2,
               const Math::SurfaceVectorField &m_c_1, const Math::SurfaceVectorField &m_c_2,
               const Containers::vector<Mesh::point_t> &points, Types::complex_d k) {
    Containers::vector<Types::Vector3c> field;
    field.resize(points.size());
#pragma omp parallel for num_threads(14) schedule(static)
    for (int i = 0; i < points.size(); ++i) {
        field[i] =
            Lattice::getE_in_point(e_c_1, m_c_1, k, points[i]) + Lattice::getE_in_point(e_c_2, m_c_2, k, points[i]);
    }
    return field;
}

void getSigmaValuesXZ(const Types::complex_d k, const Math::SurfaceVectorField &j_e_1, const Math::SurfaceVectorField &j_e_2,
                    const Math::SurfaceVectorField &j_m_1, const Math::SurfaceVectorField &j_m_2) {
    int samples = 360;
    Containers::vector<Types::scalar> esas;
    esas.reserve(samples);
    Containers::vector_d angles;
    angles.reserve(samples);

    for (int i = 0; i < samples; i++) {
        Types::scalar angle = i * Math::Constants::PI<Types::scalar>() * 2 / samples;
        Types::Vector3d tau = {std::sin(angle), 0, std::cos(angle)};
        esas.push_back(ESA::calculateESA(tau, k, {j_e_1, j_e_2}, {j_m_1, j_m_2}));
        angles.push_back(angle);
    }
    std::ofstream sigma(
        "/home/evgen/Education/MasterDegree/thesis/results/lattice/sigmaXZ.csv");

    Utils::to_csv(esas, angles, "sigma", "angle", sigma);
}

int main() {
    // считываем сетку на антенне
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/8000_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/2000_cells.csv";
    const EMW::Types::index nNodes = 8000;
    const EMW::Types::index nCells = 2000;

    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile, nNodes, nCells);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.first, parser_out.second};
    const Types::Vector3d origin_1(-0.05, 0, 0);
    const Types::Vector3d origin_2(0.05, 0, 0);
    // сетка для первой антенны
    auto mesh_all = Mesh::Utils::move_by_vector(mesh_base, origin_1);
    mesh_all.setName("antenna_1");
    auto mesh_zero_1 = mesh_all.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    mesh_zero_1.setName("zero_mesh_1");
    // сетка для второй антенны
    auto mesh_2 = Mesh::Utils::move_by_vector(mesh_base, origin_2);
    mesh_2.setName("antenna_2");
    auto mesh_zero_2 = mesh_2.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    mesh_zero_2.setName("zero_mesh_2");

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

    // тут векторное поле на активном сечении (порту) (direct wave в волноводе)
    const auto get_e_h10_mode = [&k, &a, &beta](const Mesh::point_t &x) {
        const auto pi = Math::Constants::PI<Types::scalar>();
        const Types::complex_d mult = Math::Constants::i * (a / pi) * k / Math::Constants::e_0_c;
        const Types::complex_d exp = std::exp(Math::Constants::i * beta * (x.z() + 0.17));
        const Types::scalar sin = std::cos(pi * x.x() / a);
        return Types::Vector3c{Types::complex_d{0, 0}, mult * sin * exp, Types::complex_d{0, 0}};
    };

    // собираем им матрицу
    const auto matrix = Lattice::getSLAE(mesh_all, mesh_2, a, k);

    // собираем правую часть шаманским спобосом (очень шаманским)
    const auto rhs_1 = Lattice::getRhs(mesh_base, mesh_base.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION),
                                       get_e_h10_mode);
    Types::VectorXc rhs{2 * rhs_1.rows()};
    rhs.block(0, 0, rhs_1.rows(), 1) = rhs_1;
    // добавляем фазу второй антенне
    const Types::scalar phi = 0 * Math::Constants::PI<Types::scalar>();
    const Types::complex_d phase_factor = std::exp(Math::Constants::i * phi);
    rhs.block(rhs_1.rows(), 0, rhs_1.rows(), 1) = phase_factor * rhs_1;

    std::cout << "Matrix assembled, size: " << matrix.rows() << std::endl;

    const auto result = Research::solve<Eigen::GMRES>(matrix, rhs, 1000, 1e-2);

    // Разбиваем на токи и рисуем на разных многообразиях
    const Types::VectorXc electric_current_1 = result.block(0, 0, 2 * mesh_all.getCells().size(), 1);
    const Types::VectorXc magnetic_current_1 =
        result.block(2 * mesh_all.getCells().size(), 0, 2 * mesh_zero_1.getCells().size(), 1);
    const Types::VectorXc electric_current_2 = result.block(rhs_1.rows(), 0, 2 * mesh_all.getCells().size(), 1);
    const Types::VectorXc magnetic_current_2 =
        result.block(rhs_1.rows() + 2 * mesh_all.getCells().size(), 0, 2 * mesh_zero_1.getCells().size(), 1);

    const std::string path = "/home/evgen/Education/MasterDegree/thesis/results/lattice/";

    Math::SurfaceVectorField e_c_1 = Math::SurfaceVectorField::TangentField(mesh_all, electric_current_1);
    e_c_1.setName("e_c");
    Math::SurfaceVectorField m_c_1 = Math::SurfaceVectorField::TangentField(mesh_zero_1, magnetic_current_1);
    m_c_1.setName("m_c");
    Math::SurfaceVectorField e_c_2 = Math::SurfaceVectorField::TangentField(mesh_2, electric_current_2);
    e_c_2.setName("e_c");
    Math::SurfaceVectorField m_c_2 = Math::SurfaceVectorField::TangentField(mesh_zero_2, magnetic_current_2);
    m_c_2.setName("m_c");

    VTK::united_snapshot({e_c_1}, {}, mesh_all, path);
    VTK::united_snapshot({e_c_2}, {}, mesh_2, path);

    // Рисуем картину поля в плоскости y = 0
    int N1 = 100;
    Types::scalar h1 = 1. / (N1 - 1);
    int N2 = 200;
    Types::scalar h2 = 2. / (N2 - 1);

    std::vector<Mesh::point_t> points;
    points.reserve(N1 * N2);
    Mesh::Utils::cartesian_product_unevenXZ(std::ranges::views::iota(0, N1), std::ranges::views::iota(0, N2),
                                            std::back_inserter(points), N1, N2, h1, h2);

    const auto calculated_field = calculateField(e_c_1, e_c_2, m_c_1, m_c_2, points, k);

    VTK::field_in_points_snapshot({calculated_field}, {"E"}, points, "surrounding_mesh", path);

    // считаем ДН системы из двух антенн
    getSigmaValuesXZ(k, e_c_1, e_c_2, m_c_1, m_c_2);
}
