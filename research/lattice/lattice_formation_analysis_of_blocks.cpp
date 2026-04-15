//
// Created by evgen on 06.03.2025.
//
#include <string>

#define EIGEN_USE_LAPACKE
#define EIGEN_USE_BLAS

#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/matrix/iterative_solvers_coverage/MatrixReplacement.hpp"

#include "geometry/PeriodicStructure.hpp"

#include "VTKFunctions.hpp"

#include "FieldCalculation.hpp"
#include "GeneralEquation.hpp"

#include "experiment/PhysicalCondition.hpp"

#include "Utils.hpp"

#include <chrono>
#include <iostream>

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

void experiment(const Types::MatrixXc &mat, const Types::VectorXc &vec) {
    const Eigen::JacobiSVD<Types::MatrixXc, Eigen::ComputeFullU> svd(mat);
    std::cout << "Convergence: " << svd.info() << std::endl;
    std::cout << "U computing: " << svd.computeU() << std::endl;
    const Types::VectorXc Uf = svd.matrixU().adjoint() * vec;

    const std::string path =
        "/home/evgen/Education/MasterDegree/thesis/results/lattice/spectral_analysis/matrix_rhs_consistency/";
    std::ofstream file(path + "rhs_1_1.csv");
    Utils::to_csv(svd.singularValues(), Uf, "sigmas", "components", file, ' ');
}

namespace LAMatrix = Math::LinAgl::Matrix;
// каким методом расчитываем матрицу
constexpr Research::Lattice::CalculationMethod calc_method = Research::Lattice::CalculationMethod::ACA;
using TTBMatrix = Research::Lattice::CalcTraits<calc_method>::ReturnType;

int main() {
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "open_waveguide/3466_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "open_waveguide/3434_cells.csv";

    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells, parser_out.tags};

    constexpr Types::index N1 = 1; // количество строк в решетке
    constexpr Types::index N2 = 1; // количество столбцов в решетке
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
    // 3) Собираем правую часть
    const auto rhs = WaveGuideWithActiveSection::getRhs(std::array{Types::complex_d{1., 0}, Types::complex_d{1., 0}},
                                                        mesh_base, a, k);
    std::cout << "Rhs assembled, norm = " << rhs.norm() << std::endl;

#define FULL 1
#if FULL
    // 1) Честно собираем полную матрицу
    const auto matrix = Research::Lattice::getMatrix<Research::Lattice::CalculationMethod::Full>(geometry, a, k);
    const Types::MatrixXc matrix_full = matrix.to_dense();
    std::cout << "Matrix size: " << matrix.rows() << std::endl;
#else
    // 2) собираем общую маленькую тёплицеву матрицу
    Research::Lattice::CalcTraits<calc_method>::error_controller = 0.01;
    const auto matrix = Research::Lattice::getMatrix<calc_method>(geometry, a, k);
    // Бахаем её в плотный формат
    const auto matrix_full = matrix.to_dense();
#endif

    // 4) Эксперимент
    std::cout << "Experiment started" << std::endl;
    experiment(matrix_full, rhs);
}
