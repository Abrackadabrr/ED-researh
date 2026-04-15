//
// Created by evgen on 10.09.2025.
//

#include <string>

#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "geometry/PeriodicStructure.hpp"

#include "VTKFunctions.hpp"

#include "../GeneralEquation.hpp"

#include "math/matrix/Matrix.hpp"

#include "experiment/PhysicalCondition.hpp"

#include <Utils.hpp>
#include <iostream>

namespace eq = WaveGuideWithActiveSection;
using namespace EMW;

template <int N1, int N2> using Scene = Geometry::PeriodicStructure<N1, N2>;

namespace LAMatrix = Math::LinAgl::Matrix;
using TTBMatrix = Math::LinAgl::Matrix::ToeplitzToeplitzBlock<Types::complex_d>;

template <typename block_t> TTBMatrix get_typical_matrix(Types::index N, Types::index M, const block_t &block) {
    const auto get_internal_block = [&](Types::index i, Types::index j) { return block; };

    const auto get_external_block = [&](Types::index i, Types::index j) {
        return Math::LinAgl::Matrix::ToeplitzBlock<Types::complex_d>(M, M, get_internal_block);
    };

    return TTBMatrix(N, N, get_external_block);
}

int main() {
    // считываем сетку на антенне
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "open_waveguide/200_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "open_waveguide/192_cells.csv";

    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells, parser_out.tags};

    constexpr Types::index N1 = 5; // количество строк в решетке
    constexpr Types::index N2 = 5; // количество столбцов в решетке
    constexpr Types::index N1_x_N2 = N1 * N2;

    const Types::scalar a = 0.07;
    const Types::scalar freq = Math::Constants::c / 1e8;
    const Types::complex_d k{Physics::get_k_on_frquency(freq), 0};
    const Types::complex_d beta = std::sqrt(k * k - (EMW::Math::Constants::PI_square<Types::scalar>() / (a * a)));

    const auto diagonal_block = WaveGuideWithActiveSection::diagonal(mesh_base, a, k);

    const auto matrix_ttb = get_typical_matrix(N1, N2, diagonal_block);

    std::cout << "Matrix is assembled, memory usage: " << Utils::get_memory_usage(matrix_ttb).toeplitz_matrix << "  Гб"
              << std::endl;

    const Types::VectorXc test_vec = Types::VectorX<Types::complex_d>::Random(matrix_ttb.cols());

    // const auto result = matrix_ttb * test_vec;
    const auto result_2 = matrix_ttb.matvec(test_vec);

    std::cout << result_2[0] << std::endl;
}
