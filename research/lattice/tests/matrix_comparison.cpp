//
// Created by evgen on 22.06.2025.
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

#include <Utils.hpp>
#include <chrono>
#include <experiment/ESA.hpp>
#include <iostream>

namespace eq = WaveGuideWithActiveSection;

template <int N1, int N2> using Scene = Geometry::PeriodicStructure<N1, N2>;

namespace LAMatrix = Math::LinAgl::Matrix;
using TTBMatrix = LAMatrix::ToeplitzToeplitzDynFactoredBlock<Types::complex_d>;

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
    constexpr Types::index N2 = 2;
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

    const auto &ref_mesh = geometry.get(0);
    const auto &an_mesh = geometry.get(1);
    // А теперь смотрим на сборку матрицы целиком и поэлементно
    // Собираем матрицу и смотрим время
    auto start = std::chrono::system_clock::now();

    const auto out_block = WaveGuideWithActiveSection::submatrix(ref_mesh, an_mesh, a, k);

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix assembling: " << elapsed.count() << " ms" << std::endl;

#if 0
    Types::MatrixXc out_block_element_wise = Types::MatrixXc::Zero(out_block.rows(), out_block.cols());
#pragma omp parallel for num_threads(14) collapse(2)
    for (int i = 0; i < out_block.rows(); ++i)
        for (int j = 0; j < out_block.cols(); ++j) {
            out_block_element_wise(i, j) = WaveGuideWithActiveSection::element_of_submatrix(i, j,
                ref_mesh,
                ref_mesh.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION),
                an_mesh,
                an_mesh.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION),
                a, k);

            // comparison
            const auto element_error = out_block(i, j) - out_block_element_wise(i, j);
            if (std::abs(element_error / out_block(i, j)) > 1e-15) {
                std::cout << element_error << std::endl;
                std::cout << i << ", " << j << std::endl;
                std::terminate();
            }
        }
#endif

    // проверка работы крестовой аппроксимации на нашей реальной матрице
    // размеры самых внутренних блоков матрицы (нужна для АСА)
    const Types::index N_cols = ref_mesh.getCells().size();
    const Types::index K_cols = ref_mesh.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION).getCells().size();
    const Types::index cols = 2 * (N_cols + K_cols);
    const Types::index rows = cols;

    // тут дурацкая функция для расчета элемента матрицы
    const auto &ref_mesh_zero = ref_mesh.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    const auto &another_mesh_zero = an_mesh.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    // cells extraction
    const auto & ref_cells = ref_mesh.getCells();
    const auto & ref_zero_cells = ref_mesh_zero.getCells();
    const auto & an_cells = an_mesh.getCells();
    const auto & an_zero_cells = another_mesh_zero.getCells();
    const auto element_function = [ref_cells, ref_zero_cells, an_cells, an_zero_cells, k,
                                   a](Types::index row, Types::index col) -> Types::complex_d {
        return WaveGuideWithActiveSection::element_of_submatrix(row, col, ref_cells, ref_zero_cells,
                                                                an_cells, an_zero_cells, a, k);
    };

    const auto hack_element_function = [&out_block](Types::index i, Types::index j) { return out_block(i, j); };

    // тут непосредственно делаем адаптивный крест
    start = std::chrono::system_clock::now();

    const auto result =
        Math::LinAgl::Decompositions::ComplexACA::compute(element_function, out_block.rows(), out_block.cols(), 0.1);

    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Elapsed time for ACA: " << elapsed.count() << " ms" << std::endl;
#if 1
    const auto approximation = result.compute();
    const Types::MatrixXc error_matrix = (out_block - approximation);
    std::cout << "Rank = " << result.get<0>().cols() << std::endl;
    std::cout << "Absolute error of approximation = " << error_matrix.norm() << std::endl;
    std::cout << "Relative error of approximation = " << error_matrix.norm()  / out_block.norm() << std::endl;
#endif
}
