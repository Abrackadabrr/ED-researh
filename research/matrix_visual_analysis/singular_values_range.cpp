//
// Created by evgen on 09.06.2025.
//

#include <string>

#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/matrix/iterative_solvers_coverage/MatrixReplacement.hpp"

#include "geometry/PeriodicStructure.hpp"

#include "VTKFunctions.hpp"

#include "../lattice/FieldCalculation.hpp"
#include "../lattice/GeneralizedEquations.hpp"

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


void dump_matrix(const Mesh::SurfaceMesh& mesh_base, const Types::scalar d, int number) {
    // путь до папки, куда писать результаты
    const std::string path = "/home/evgen/Education/MasterDegree/thesis/results/lattice/spectral_analysis/range_of_svds/";

    // сделаем ещё одну сетку для анализа внедиагонлаьных элементов
    const auto &mesh_1 = mesh_base;

    const Types::Vector3d origin{d, 0, 0.0};
    const auto mesh_2 = Mesh::Utils::move_by_vector(mesh_base, origin);

    // Геометрические параметры антенн
    // Короткая сторона волновода
    const Types::scalar a = 0.07;
    // Физика волны в пространстве
    // частота в гигагерцах
    const Types::scalar freq = 2 * Math::Constants::c / 1e8;
    const Types::complex_d k{Physics::get_k_on_frquency(freq), 0};

    const auto out_diagonal_block = WaveGuideWithActiveSection::submatrix(mesh_2, mesh_1, a, k);

    const auto svd = out_diagonal_block.bdcSvd();

    const auto sing_vals = svd.singularValues();
    Containers::vector<int> indexes{};
    indexes.resize(sing_vals.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    // Запись сингулярных чисел
    std::ofstream file(path + "sv_out" + std::to_string(number) +".csv");
    Utils::to_csv(indexes, sing_vals, "ind", "sv", file);
}

int main() {
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/8000_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/2000_cells.csv";
    constexpr EMW::Types::index nNodes_c = 8000;
    constexpr EMW::Types::index nCells_c = 2000;

    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.first, parser_out.second};

    std::vector<Types::scalar> d(100); std::iota(d.begin(), d.end(), 0);
    const Types::scalar a = 0.07;
    std::transform(d.begin(), d.end(), d.begin(), [&](const Types::scalar value) {return value < 10 ? value * a / 10 : a + (a / 5) * (value - 10);});
    std::copy(d.begin(), d.end(), std::ostream_iterator<Types::scalar>(std::cout, " "));
#pragma omp parallel for num_threads(10) schedule(dynamic)
    for (int i = 0; i < d.size(); ++i) {
        dump_matrix(mesh_base, d[i], i);
        std::cout << i << " done, d = " << d[i]  << std::endl;
    }
}
