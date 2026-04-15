//
// Created by evgen on 02.06.2025.
//
#include <string>

#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/matrix/iterative_solvers_coverage/MatrixReplacement.hpp"

#include "geometry/PeriodicStructure.hpp"

#include "../lattice/GeneralEquation.hpp"

#include "VTKFunctions.hpp"

#include "../lattice/FieldCalculation.hpp"
#include "../lattice/SpecificLatticeEquations.hpp"

#include "experiment/PhysicalCondition.hpp"

#include "Utils.hpp"

#include <chrono>
#include <iostream>
#include <numeric>

template <typename Matrix> void dump(const Matrix &matrix, const std::string &filename) {
    std::ofstream file(filename);
    file.precision(20);
    for (Types::index row = 0; row < matrix.rows(); ++row) {
        for (Types::index col = 0; col < matrix.cols(); ++col) {
            file << matrix(row, col).real() << "+" << matrix(row, col).imag() << "j";
            if (col !=  matrix.cols() - 1) file << " ";
        }
        file << std::endl;
    }
    file.close();
}

namespace LAMatrix = Math::LinAgl::Matrix;
// каким методом расчитываем матрицу
constexpr Research::Lattice::CalculationMethod calc_method = Research::Lattice::CalculationMethod::Full;
using TTBMatrix = Research::Lattice::CalcTraits<calc_method>::ReturnType;

int main() {
    const std::string nodesFile_c = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/open_waveguide/72_nodes.csv";
    const std::string cellsFile_c = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/open_waveguide/67_cells.csv";
    constexpr EMW::Types::index nNodes_c = 8000;
    constexpr EMW::Types::index nCells_c = 2000;

    // путь до папки, куда писать результаты
    const std::string path = "/home/evgen/Education/MasterDegree/thesis/results/blocks_analysis/";
    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile_c, cellsFile_c);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells, parser_out.tags};

    constexpr Types::index N1 = 2;  // количество строк в решетке
    constexpr Types::index N2 = 3;  // количество столбцов в решетке
    constexpr Types::index N1_x_N2 = N1 * N2;
    constexpr Types::scalar d1 = 0.07;  // расстояние между строками
    constexpr Types::scalar d2 = 0.14;   // расстояние между столбцами
    const Types::Vector3d dir1 = Types::Vector3d{0, -1, 0}.normalized();
    const Types::Vector3d dir2 = Types::Vector3d{1, 0, 0}.normalized();
    const Geometry::PeriodicStructure<N1, N2> geometry{dir1, dir2, d1, d2, mesh_base};

    // Геометрические параметры антенн
    // Короткая сторона волновода
    const Types::scalar a = 0.07;
    // Физика волны в пространстве
    // частота в гигагерцах
    const Types::scalar freq = 2 * Math::Constants::c / 1e8;
    const Types::complex_d k{Physics::get_k_on_frquency(freq), 0};

    // расчет матрицы
    const auto matrix = Research::Lattice::getMatrix<calc_method>(geometry, a, k).to_dense();

    VTK::geometry_snapshot(geometry, "/home/evgen/Education/MasterDegree/thesis/results/blocks_analysis/geom.vtu");

    dump(matrix, "/home/evgen/Education/MasterDegree/thesis/results/blocks_analysis/TTB_example.txt");
}
