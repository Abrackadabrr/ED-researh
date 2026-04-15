//
// Created by evgen on 03.04.2025.
//

#include <string>

#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"


#include "math/matrix/iterative_solvers_coverage/MatrixReplacement.hpp"
#include "math/matrix/iterative_solvers_coverage/MatrixTraits.hpp"

#include "research/Solve.hpp"

#include "geometry/PeriodicStructure.hpp"

#include "VTKFunctions.hpp"

#include "FieldCalculation.hpp"
#include "GeneralEquation.hpp"
#include "SpecificLatticeEquations.hpp"

#include "math/matrix/Matrix.hpp"

#include <unsupported/Eigen/IterativeSolvers>

#include "experiment/PhysicalCondition.hpp"

#include <Utils.hpp>
#include <chrono>
#include <equations/EquationsOverGeometry.hpp>
#include <experiment/ESA.hpp>
#include <iostream>

namespace eq = WaveGuideWithActiveSection;

template <int N1, int N2> using Scene = Geometry::PeriodicStructure<N1, N2>;
namespace LAMatrix = Math::LinAgl::Matrix;
using TTBMatrix = LAMatrix::ToeplitzToeplitzBlock<Types::complex_d>;

int main() {
    // считываем сетку на антенне
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/8000_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/2000_cells.csv";

    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.first, parser_out.second};

    constexpr Types::index N1 = 1;
    constexpr Types::index N2 = 3;
    constexpr Types::index N1_x_N2 = N1 * N2;
    const Scene<N1, N2> geometry{{1, 0, 0}, {0, 1, 0},0.1, 0.07, mesh_base};

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

    // собираем тёплицеву матрицу
    const auto matrix = Research::Lattice::getMatrix<Research::Lattice::CalculationMethod::Full>(geometry, a, k);
    std::cout << "Toeplitz matrix is ready" << std::endl;

    // собираем общую матрицу (большую и плотную!)
    const auto matrix_full = Equations::MatrixForStructure::compute(geometry, eq::diagonal, eq::submatrix, {a, k});
    std::cout << "Full matrix is ready" << std::endl;

    // собираем факторизованную матрицу
//    const auto matrix_factored =
//        Research::Lattice::getMatrix<Research::Lattice::CalculationMethod::Compressed>(geometry, a, k);
    std::cout << "Factored matrix is ready" << std::endl;

    // Считаем ошибку
    std::cout << (matrix.to_dense() - matrix_full).norm() << std::endl;
//    std::cout << (matrix_full - matrix_factored.to_dense()).norm() << std::endl;

    // Проверка умножения на матрицу
    for (int i = 0; i < 100; ++i) {
        const Types::VectorXc vec = Types::VectorXc::Random(matrix.rows());
        const auto res_toeplitz = matrix * vec;
        const Types::VectorXc res_full = matrix_full * vec;
        std::cout << "----- Iteration " << i << " -----" << std::endl;
        std::cout << (res_toeplitz - res_full).norm() << std::endl;
        std::cout << (res_toeplitz - res_full).norm() / res_full.norm() << std::endl;
    }
}
