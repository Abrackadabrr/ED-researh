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

#include "geometry/HomogeneousStructure.hpp"

#include "VTKFunctions.hpp"

#include "../GeneralEquation.hpp"
#include "../FieldCalculation.hpp"
#include "../FieldOverGeometry.hpp"

#include "math/matrix/Matrix.hpp"

#include <unsupported/Eigen/IterativeSolvers>

#include "experiment/PhysicalCondition.hpp"

#include <Utils.hpp>
#include <chrono>
#include <equations/EquationsOverGeometry.hpp>
#include <experiment/ESA.hpp>
#include <iostream>

namespace eq = WaveGuideWithActiveSection;

using Scene = Geometry::HomogeneousStructure;

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
using matrix = Types::MatrixXc;


int main() {
    // считываем сетку на антенне
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/8000_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/2000_cells.csv";
    constexpr EMW::Types::index nNodes = 8000;
    constexpr EMW::Types::index nCells = 2000;

    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.first, parser_out.second};
    // собираем orinigs
    const Types::scalar a_hat = 0.04;  // расстояние между центрами сеток "на диагонали 1"
    const Types::scalar b_hat = 0.08;  // расстояние между центрами сеток "на диагонали 2"
    // 2 3 2 сетка
    const Containers::vector<Types::Vector3d> origins{
            Types::Vector3d{-b_hat / 2, -a_hat, 0}, Types::Vector3d{b_hat / 2, -a_hat, 0},
    Types::Vector3d{-b_hat, 0, 0}, Types::Vector3d{0, 0, 0}, Types::Vector3d{b_hat, 0, 0},
            Types::Vector3d{-b_hat / 2, a_hat, 0}, Types::Vector3d{b_hat / 2, a_hat, 0}};

    const Scene geometry{origins, mesh_base};

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

    // собираем общую самую большую в мире матрицу

    auto start = std::chrono::high_resolution_clock::now();

    const auto matrix = Equations::MatrixForStructure::compute(geometry, WaveGuideWithActiveSection::diagonal, WaveGuideWithActiveSection::submatrix, {a, k});

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    // собираем правую часть шаманским способом (очень шаманским)
    // решаем какой будет фазовый фактор на волноводах
    const Containers::array<Types::scalar, 7> phases{0};
    Containers::array<Types::complex_d, 7> phase_factors;
    for (Types::index i = 0; i < phases.size(); ++i)
        phase_factors[i] = std::exp(Math::Constants::i * phases[i] * Math::Constants::deg_to_rad<Types::scalar>());
    const auto rhs = eq::getRhs(phase_factors, mesh_base, a, k);

    std::cout << "RHS assembled, size: " << rhs.rows() << std::endl;

    auto result = Research::solve<Eigen::GMRES>(matrix, rhs, 2000, 1e-2);

    // Разбиваем на токи и рисуем на разных многообразиях
    const Research::Lattice::FieldOver field_set(geometry, std::move(result));

    const std::string path = "/home/evgen/Education/MasterDegree/thesis/results/fal/super_full/";
    const std::string dir_name = "FAL/";
    VTK::set_of_fields_snapshot(field_set, path + dir_name + "2_3_2_lattice.vtu");

    getSigmaValuesXZ(k, field_set, path + dir_name);
    getSigmaValuesYZ(k, field_set, path + dir_name);
}
