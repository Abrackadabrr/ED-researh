//
// Created by evgen on 31.01.2026.
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

#include "FieldCalculation.hpp"
#include "FieldOverGeometry.hpp"
#include "GeneralEquation.hpp"

#include "math/matrix/Matrix.hpp"

#include <unsupported/Eigen/IterativeSolvers>

#include "experiment/PhysicalCondition.hpp"

#include <Utils.hpp>
#include <chrono>
#include <experiment/ESA.hpp>
#include <iostream>

namespace eq = WaveGuideWithActiveSection;

template <int N1, int N2> using Scene = Geometry::PeriodicStructure<N1, N2>;

template <typename Matrix> void dump(const Matrix &matrix, const std::string &filename) {
    std::ofstream file(filename);
    file.precision(20);
    for (Types::index row = 0; row < matrix.rows(); ++row) {
        for (Types::index col = 0; col < matrix.cols(); ++col) {
            file << matrix(row, col).real() << "+" << matrix(row, col).imag() << "j";
            if (col != matrix.cols() - 1)
                file << " ";
        }
        file << std::endl;
    }
    file.close();
}

namespace LAMatrix = Math::LinAgl::Matrix;
// каким методом расчитываем матрицу
constexpr Research::Lattice::CalculationMethod calc_method = Research::Lattice::CalculationMethod::Full;
using TTBMatrix = Research::Lattice::CalcTraits<calc_method>::ReturnType;
// предобуславливание
using DiagonalPrec = LAMatrix::Preconditioning::DiagonalPreconditioner<Types::complex_d, TTBMatrix>;
using BlockDiagPrec = LAMatrix::Preconditioning::BlockDiagonalPreconditioner<Types::complex_d, TTBMatrix>;
using NoPrec = LAMatrix::Preconditioning::IdentityPreconditioner<Types::complex_d, TTBMatrix>;
using MatrixWrapper = LAMatrix::Wrappers::MatrixReplacement<TTBMatrix, DiagonalPrec>;

int main() {
    // считываем сетку на антенне
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "open_waveguide/648_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "open_waveguide/634_cells.csv";

    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells, parser_out.tags};

    constexpr Types::index N1 = 2; // количество строк в решетке
    constexpr Types::index N2 = 2; // количество столбцов в решетке
    constexpr Types::index N1_x_N2 = N1 * N2;
    constexpr Types::scalar d1 = 0.04; // расстояние между строками
    constexpr Types::scalar d2 = 0.08; // расстояние между столбцами
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

    // собираем общую маленькую тёплицеву матрицу
    // Research::Lattice::CalcTraits<calc_method>::error_controller = 0.01;

    auto start = std::chrono::high_resolution_clock::now();

    const auto matrix = Research::Lattice::getMatrix<calc_method>(geometry, a, k);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << Utils::get_memory_usage(matrix) << std::endl;
    std::cout << "Matrix assembled, size: " << matrix.rows() << "; time elapsed: " << elapsed << std::endl;

    const auto dense_matrix = matrix.to_dense();
    auto prec_matrix = dense_matrix;
    // std::string dir_path = "/home/evgen/Education/MasterDegree/thesis/results/preconditioning_periodic_structure/";
    // dump(dense_matrix, dir_path + "simple_mat.csv");

    //  теперь соберем диагональный предобуславливатель к этой матрице
    const Types::MatrixXc diagonal_block = matrix.get_block(0, 0).get_block(0,0); //.compute();
    const Types::MatrixXc inverse_diagonal = diagonal_block.inverse();
    std::cout << (diagonal_block * inverse_diagonal - Types::MatrixXc::Identity(diagonal_block.rows(), diagonal_block.cols())).norm() << std::endl;
    for (int i = 0; i < N1_x_N2; ++i) {
          prec_matrix.block(0,i * inverse_diagonal.rows(), dense_matrix.rows(), inverse_diagonal.cols()) =
               prec_matrix.block(0,i * inverse_diagonal.rows(), dense_matrix.rows(), inverse_diagonal.cols()) * inverse_diagonal;
    }
    // dump(dense_matrix, dir_path + "prec_mat.csv");

    // теперь сделаем решение с обеими матрицами GMRESом
    // собираем правую часть шаманским способом
    // решаем какой будет фазовый фактор на волноводах
    const Containers::array<Types::scalar, N1_x_N2> phases{0};
    Containers::array<Types::complex_d, N1_x_N2> phase_factors;
    for (Types::index i = 0; i < phases.size(); ++i)
        phase_factors[i] = std::exp(Math::Constants::i * 0. * Math::Constants::deg_to_rad<Types::scalar>());
    const Types::VectorXc rhs = Types::VectorXc::Random(dense_matrix.rows());

    std::cout << "RHS assembled, size: " << rhs.rows() << "; rhs norm = " << rhs.norm() << std::endl;

    // решалка номер раз
    const auto res_dense = Research::solve<Eigen::GMRES>(dense_matrix, rhs, 5000, 1e-4);
    // решалка номер два
    auto res_prec = Research::solve_without_prec<Eigen::GMRES>(prec_matrix, rhs, 2000, 1e-4);
    res_prec = (inverse_diagonal * res_prec.reshaped(diagonal_block.rows(), N1_x_N2)).reshaped(res_prec.rows(), 1);

    std::cout << "Rel err = " << (res_prec - res_dense).norm() / res_prec.norm() << std::endl;
}
