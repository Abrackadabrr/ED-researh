//
// Created by evgen on 09.09.2025.
//

#include "geometry/PeriodicStructure.hpp"

#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/matrix/Matrix.hpp"
#include "math/matrix/decompositions/Decompositions.hpp"

#include "../SpecificLatticeEquations.hpp"

#include "experiment/PhysicalCondition.hpp"

#include <iostream>
#include <string>

namespace eq = WaveGuideWithActiveSection;
using namespace EMW;

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

    const auto &ref_mesh_zero = ref_mesh.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    const auto &another_mesh_zero = an_mesh.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    const auto &ref_cells = ref_mesh.getCells();
    const auto &ref_zero_cells = ref_mesh_zero.getCells();
    const auto &an_cells = an_mesh.getCells();
    const auto &an_zero_cells = another_mesh_zero.getCells();
    Types::index N = 2 * (ref_cells.size() + ref_zero_cells.size());

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

// main function to bench
    const auto result =
            Math::LinAgl::Decompositions::ComplexACA::compute(compute_row, compute_col, N, N, 0.1);

    std::cout << "Rank = " << result.get<0>().cols() << std::endl;
}
