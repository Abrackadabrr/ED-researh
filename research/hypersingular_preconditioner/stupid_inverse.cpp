//
// Created by evgen on 08.08.2025.
//

#include "DiscreteLaplacian.hpp"
#include "OperatorT_simple.hpp"

#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"

#include "visualisation/VTKFunctions.hpp"

#include "Solve.hpp"

#include "unsupported/Eigen/IterativeSolvers"

#include <string>

using namespace EMW;

int main() {
    std::vector<int> Ns{10, 20, 30, 40, 50, 100};
    for(auto&& N_POINTS : Ns) {
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

        // дискретизация гиперсингулярного интеграла по заданной сетке
        const auto T_matrix = EMW::OperatorT::LaplaceEquation::T_over_mesh(mesh, mesh);

        // собираем проектор
        const auto P = DiscreteLaplacian::getBoundaryAnnulator(N_POINTS, N_POINTS);

        // cбор дискретного оператора лапласа
        const auto laplacian =
            DiscreteLaplacian::discreteLaplacian(N_POINTS + 2, N_POINTS + 2, 2. / (N_POINTS - 1), 2. / (N_POINTS - 1));

        // сбор предобуславливателя непосредственно
        const Types::MatrixXd explicit_preconditioner = Types::MatrixXd(laplacian).inverse() * T_matrix;

        // предобусловленная матрица
        const Types::MatrixXd new_matrix = T_matrix * explicit_preconditioner;

        // считаем новое число обсуловленности для матрицы
        const auto new_mat_svd = new_matrix.bdcSvd();
        const auto new_mat_sing_vals = new_mat_svd.singularValues();
        std::cout << "Размер сетки N = " << N_POINTS << std::endl;
        std::cout << "Границы спектра предобусловенной матрицы матрицы: " << new_mat_sing_vals[0] << ", " << new_mat_sing_vals[new_mat_sing_vals.size() - 1] << std::endl;
        std::cout << "Спектральное число обусловленности: " << new_mat_sing_vals[0] / new_mat_sing_vals[new_mat_sing_vals.size() - 1] << std::endl << std::endl;
    }
}
