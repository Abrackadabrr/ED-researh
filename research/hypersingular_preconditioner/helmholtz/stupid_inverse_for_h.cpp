//
// Created by evgen on 08.08.2025.
//

#include "../DiscreteLaplacian.hpp"
#include "../OperatorT_simple.hpp"

#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"

#include "visualisation/VTKFunctions.hpp"

#include "../Solve.hpp"

#include "unsupported/Eigen/IterativeSolvers"

#include <research/hypersingular_preconditioner/OperatorT_simple.hpp>
#include <string>

using namespace EMW;

int main() {
    int N_POINTS = 50;

    const std::string file_nodes =
        "/home/evgen/Education/Schools/Sirius2025/meshes/" + std::to_string(N_POINTS) + "/nodes.csv";
    const std::string file_cells =
        "/home/evgen/Education/Schools/Sirius2025/meshes/" + std::to_string(N_POINTS) + "/cells.csv";
    const std::string path_to_res = "/home/evgen/Education/Schools/Sirius2025/results/";

    // делаем сетку
    const auto [nodes, cells] = EMW::Parser::parseMesh(file_nodes, file_cells);
    auto mesh = EMW::Mesh::SurfaceMesh(nodes, cells);
    const Types::index n_cells = mesh.getCells().size();
    // ищем граничные узлы
    auto boundary_cells_indexes = DiscreteLaplacian::detail::get_boundary_cells_indexes(N_POINTS);
    for (int i = 0; i < mesh.getCells().size(); i++) {
        if (boundary_cells_indexes.contains(i))
            mesh.getCells()[i].tag = Mesh::IndexedCell::Tag::BOUNDARY_CELL;
        else
            mesh.getCells()[i].tag = Mesh::IndexedCell::Tag::INTERNAL_CELL;
    }

    const Types::complex_d k{20, 0};
    const auto l = 2 * Math::Constants::PI<Types::scalar>() / k.real();
    std::cout << "Длина волны = " << l << std::endl;
    std::cout << "Ячеек на длину волны = " << l * (N_POINTS - 1) / 2 << std::endl;
    // дискретизация гиперсингулярного интеграла по заданной сетке
    const auto T_matrix = EMW::OperatorT::HelmholtzEquation::T_over_mesh(k, mesh, mesh);

    // cбор дискретного оператора лапласа
    const auto helm =
        DiscreteLaplacian::discreteHelmholtz(k, N_POINTS + 2, N_POINTS + 2, 2. / (N_POINTS - 1), 2. / (N_POINTS - 1));

    // сбор предобуславливателя непосредственно
    const Types::MatrixXc explicit_preconditioner = Types::MatrixXc(helm).inverse() * T_matrix;

    // предобусловленная матрица
    const Types::MatrixXc new_matrix = T_matrix * explicit_preconditioner;

    // считаем старое и новое число обсуловленности для матрицы
    const auto T_svd = T_matrix.bdcSvd();
    const auto T_sing_vals = T_svd.singularValues();
    std::cout << "Уравнение Гельмгольца, N = " << N_POINTS << std::endl;
    std::cout << "Границы спектра исходной матрицы матрицы: " << T_sing_vals[0] << ", "
              << T_sing_vals[T_sing_vals.size() - 1] << std::endl;
    std::cout << "Спектральное число обусловленности: " << T_sing_vals[0] / T_sing_vals[T_sing_vals.size() - 1]
              << std::endl;

    const auto new_mat_svd = new_matrix.bdcSvd();
    const auto new_mat_sing_vals = new_mat_svd.singularValues();
    std::cout << "Границы спектра предобусловенной матрицы матрицы: " << new_mat_sing_vals[0] << ", " << new_mat_sing_vals[T_sing_vals.size() - 1] << std::endl;
    std::cout << "Спектральное число обусловленности: " << new_mat_sing_vals[0] / new_mat_sing_vals[new_mat_sing_vals.size() - 1] << std::endl;
}
