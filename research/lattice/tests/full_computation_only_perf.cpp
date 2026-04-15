//
// Created by evgen on 10.09.2025.
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
                                  "open_waveguide/200_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "open_waveguide/192_cells.csv";
    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells, parser_out.tags};

    // Необходимые коэффициенты для расчета
    const Types::scalar a = 0.07;
    const Types::scalar freq = Math::Constants::c / 1e8;
    const Types::complex_d k{Physics::get_k_on_frquency(freq), 0};
    const Types::complex_d beta = std::sqrt(k * k - (EMW::Math::Constants::PI_square<Types::scalar>() / (a * a)));

    const auto &ref_mesh = mesh_base;
    const auto &an_mesh = Mesh::Utils::move_by_vector(ref_mesh, {0.14, 0, 0});

    // main function to bench
    const auto result = WaveGuideWithActiveSection::submatrix(ref_mesh, an_mesh, a, k);

    std::cout << "Rank = " << result.cols() << std::endl;

}
