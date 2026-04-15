//
// Created by evgen on 09.09.2025.
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
    const auto parser_out = EMW::Parser::parse_mesh_without_tag(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells};

    // Необходимые коэффициенты для расчета
    const Types::scalar a = 0.07;
    const Types::scalar freq = Math::Constants::c / 1e8;
    const Types::complex_d k{Physics::get_k_on_frquency(freq), 0};
    const Types::complex_d beta = std::sqrt(k * k - (EMW::Math::Constants::PI_square<Types::scalar>() / (a * a)));

    const auto &ref_mesh = mesh_base;
    const auto &an_mesh = Mesh::Utils::move_by_vector(ref_mesh, {0.14, 0, 0});

    // 1) Бенчмаркинг расчета одного блока целиком
    Types::index n_repeats = 10;
    Types::scalar time_1 = 0;
    for (auto i = 0; i < n_repeats; ++i) {
        auto start = std::chrono::system_clock::now();

        const auto out_block = WaveGuideWithActiveSection::submatrix(ref_mesh, an_mesh, a, k);

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        time_1 += elapsed.count();
    }
    const auto out_block = WaveGuideWithActiveSection::submatrix(ref_mesh, an_mesh, a, k);
    std::cout << "Block full assembling: " << time_1 / n_repeats << " ms" << std::endl;

    // 2) Аналогичный бенчмаркин для креста разного типа
    const auto &ref_mesh_zero = ref_mesh.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    const auto &another_mesh_zero = an_mesh.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    const auto &ref_cells = ref_mesh.getCells();
    const auto &ref_zero_cells = ref_mesh_zero.getCells();
    const auto &an_cells = an_mesh.getCells();
    const auto &an_zero_cells = another_mesh_zero.getCells();

    const auto element_function = [ref_cells, ref_zero_cells, an_cells, an_zero_cells, k,
                                   a](Types::index row, Types::index col) -> Types::complex_d {
        return WaveGuideWithActiveSection::element_of_submatrix(row, col, ref_cells, ref_zero_cells, an_cells,
                                                                an_zero_cells, a, k);
    };

    const auto compute_row = [&ref_cells, &ref_zero_cells, &an_cells, &an_zero_cells, a, k](Types::index i) {
        return WaveGuideWithActiveSection::row_of_submatrix(i, an_cells, an_zero_cells, ref_cells, ref_zero_cells, a,
                                                            k);
    };

    const auto compute_col = [&ref_cells, &ref_zero_cells, &an_cells, &an_zero_cells, a, k](Types::index i) {
        return WaveGuideWithActiveSection::col_of_submatrix(i, an_cells, an_zero_cells, ref_cells, ref_zero_cells, a,
                                                            k);
    };

    Types::scalar error_controller = 1;

    // бенч на крест с функцией доступа к элементу
    Types::index n_cross_repeats = 50;
    Types::scalar time_2 = 0;
    for (auto i = 0; i < n_cross_repeats; ++i) {
        auto start = std::chrono::system_clock::now();

        const auto result = Math::LinAgl::Decompositions::ComplexACA::compute(element_function, out_block.rows(),
                                                                              out_block.cols(), error_controller);
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        time_2 += elapsed.count();
    }
    std::cout << "Elemented ACA assembling: " << time_2 / n_cross_repeats << " ms" << std::endl;

    // бенч на крест с расчетом строки и столбца
    Types::scalar time_3 = 0;
    for (auto i = 0; i < n_cross_repeats; ++i) {
        auto start = std::chrono::system_clock::now();

        const auto result = Math::LinAgl::Decompositions::ComplexACA::compute(compute_row, compute_col, out_block.rows(),
                                                                              out_block.cols(), error_controller);
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        time_3 += elapsed.count();
    }
    std::cout << "R-C ACA assembling: " << time_3 / n_cross_repeats << " ms" << std::endl;

    const auto result = Math::LinAgl::Decompositions::ComplexACA::compute(compute_row, compute_col, out_block.rows(),
                                                                      out_block.cols(), error_controller);

    // доп информация
    std::cout << "Rank = " << result.get<0>().cols() << std::endl;
}
