//
// Created by evgen on 01.04.2025.
//

#include <string>

#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "geometry/PeriodicStructure.hpp"

#include "GeneralEquation.hpp"

#include "math/matrix/Matrix.hpp"

#include "experiment/PhysicalCondition.hpp"

#include "Utils.hpp"

#include <chrono>
#include <iostream>

namespace eq = WaveGuideWithActiveSection;
using namespace EMW;

template <int N1, int N2> using Scene = Geometry::PeriodicStructure<N1, N2>;

namespace LAMatrix = Math::LinAgl::Matrix;
// каким методом расчитываем матрицу
constexpr Research::Lattice::CalculationMethod calc_method = Research::Lattice::CalculationMethod::ACA;
using TTBMatrix = Research::Lattice::CalcTraits<calc_method>::ReturnType;

Types::index calculate_ranks(const TTBMatrix &m) {
    Types::index sum_of_ranks = 0;
    // цикл по внешнему уровню тёплицевости
    for (int e_row = 0; e_row < m.rows_in_toeplitrz(); ++e_row) {
        for (int e_col = 0; e_col < m.cols_in_toeplitrz(); ++e_col) {
            const auto &external_block = m.get_block(e_row, e_col);
            // цикл по внутреннему уровню тёплицевости
            for (int i_row = 0; i_row < external_block.rows_in_toeplitrz(); ++i_row) {
                for (int i_col = 0; i_col < external_block.cols_in_toeplitrz(); ++i_col) {
                    const auto &internal_block = external_block.get_block(i_row, i_col);
                    if (i_col != i_row) {
                        // только внедиагональные блоки
                        sum_of_ranks += internal_block.get<0>().cols();
                        std::cout << internal_block.get<0>().cols() << " ";
                    }
                }
            }
        }
    }
    std::cout << std::endl;
    return sum_of_ranks;
}

template <typename matrix_t>
void matvec_time(const matrix_t &toeplitz) {

    Types::index n_repeat = 10;
    Types::scalar time_1 = 0;
    Types::scalar time_2 = 0;

    for (int i = 0; i < 10 * n_repeat; i++) {

        const Types::VectorX<Types::complex_d> vec = Types::VectorX<Types::complex_d>::Random(toeplitz.rows());

        auto start = std::chrono::system_clock::now();

        const Types::VectorX<Types::complex_d> result_special = toeplitz * vec;

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        time_2 += elapsed.count();
    }
    std::cout << "Произведение с новой матрицей: " << time_2 / (10 * n_repeat) << '\n';
}

int main() {
    // считываем сетку на антенне
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/8000_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "lattice/2000_cells.csv";

    // собираем сетки
    const auto parser_out = EMW::Parser::parse_mesh_without_tag(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells};

    constexpr Types::index N1 = 9;  // количество строк в решетке
    constexpr Types::index N2 = 7;  // количество столбцов в решетке
    constexpr Types::index N1_x_N2 = N1 * N2;
    constexpr Types::scalar d1 = 0.04;  // расстояние между строками
    constexpr Types::scalar d2 = 0.08;   // расстояние между столбцами
    const Types::Vector3d dir1 = Types::Vector3d{0, -1, 0}.normalized();
    const Types::Vector3d dir2 = Types::Vector3d{1, 0, 0}.normalized();
    const Geometry::PeriodicStructure<N1, N2> geometry{dir1, dir2, d1, d2, mesh_base};

    const Types::scalar a = 0.07;
    const Types::scalar freq = Math::Constants::c / 1e8;
    const Types::complex_d k{Physics::get_k_on_frquency(freq), 0};
    const Types::complex_d beta = std::sqrt(k * k - (EMW::Math::Constants::PI_square<Types::scalar>() / (a * a)));

    // эн раз замеряем время для сбора сжато матрицы
    std::vector<Types::scalar> error_controllers{0.1, 0.01, 0.001};
    for (auto &&e : error_controllers) {
        // ставим параметр контроля ошибки в аса
        Research::Lattice::CalcTraits<calc_method>::error_controller = e;

        std::cout << "// --- Эксперимент на сжатой матрицы с точностью e = "
                  << Research::Lattice::CalcTraits<calc_method>::error_controller << " ---- // " << std::endl;

        // замеряем время на сбор аппроксимации
        Types::scalar approximation_calculation_time = 0;
        Types::index n_repeats = 3;

        for (int i = 0; i < n_repeats; i++) {
            auto start = std::chrono::steady_clock::now();
            const auto matrix = Research::Lattice::getMatrix<calc_method>(geometry, a, k);

            auto end = std::chrono::steady_clock::now();
            approximation_calculation_time += std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
            std::cout << i << " iteration, " << matrix.rows() << std::endl;
        }

        const auto matrix = Research::Lattice::getMatrix<calc_method>(geometry, a, k);

        std::cout << Utils::get_memory_usage(matrix) << std::endl;
        std::cout << "Matrix assembled, size: " << matrix.rows()
                  << ";  average calculation time: " << approximation_calculation_time / n_repeats << std::endl;

        // 3) Калькулируем суммартный ранг внешних блоков
        const auto sum_of_ranks = calculate_ranks(matrix);

        std::cout << "Overall out-diagonal blocks's sum of ranks = " << sum_of_ranks << std::endl;
        std::cout << "Maximum theoretical increase = " << 1. / (1./(N1 * N2) + (2. * sum_of_ranks / (N1 * N2 * matrix.cols()))) << std::endl;

        matvec_time(matrix);
    }
}
