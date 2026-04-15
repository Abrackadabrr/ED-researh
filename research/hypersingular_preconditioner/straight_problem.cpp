//
// Created by evgen on 06.08.2025.
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
    int N_POINTS = 100;

    const std::string file_nodes =
        "/home/evgen/Education/Schools/Sirius2025/meshes/" + std::to_string(N_POINTS) + "/nodes.csv";
    const std::string file_cells =
        "/home/evgen/Education/Schools/Sirius2025/meshes/" + std::to_string(N_POINTS) + "/cells.csv";
    const std::string path_to_res = "/home/evgen/Education/Schools/Sirius2025/results/";

    // делаем сетку
    const auto [nodes, cells] = EMW::Parser::parseMesh(file_nodes, file_cells);
    auto mesh = EMW::Mesh::SurfaceMesh(nodes, cells);
    mesh.setName("Laplace");
    const Types::index n_cells = mesh.getCells().size();
    // ищем граничные узлы
    auto boundary_cells_indexes = DiscreteLaplacian::detail::get_boundary_cells_indexes(N_POINTS);
    for (int i = 0; i < mesh.getCells().size(); i++) {
        if (boundary_cells_indexes.contains(i))
            mesh.getCells()[i].tag = Mesh::IndexedCell::Tag::BOUNDARY_CELL;
        else
            mesh.getCells()[i].tag = Mesh::IndexedCell::Tag::INTERNAL_CELL;
    }

    // делаем поле правой части на сетке
    const auto rhs_field =
        Math::SurfaceScalarField<Types::scalar>::fromSLAESolution(mesh, Types::VectorXd::Ones(n_cells));
    const auto rhs = rhs_field.formVector();

    // дискретизация гиперсингулярного интеграла по заданной сетке
    const auto T_matrix = EMW::OperatorT::LaplaceEquation::T_over_mesh(mesh, mesh);
    // решение прямой задачи
    const auto res = Research::solve_without_precond<Eigen::GMRES>(T_matrix, rhs, 1000, 1e-8);

    // cбор дискретного оператора лапласа
    const auto laplacian =
        DiscreteLaplacian::discreteLaplacian(N_POINTS + 2, N_POINTS + 2, 2. / (N_POINTS - 1), 2. / (N_POINTS - 1));

    // решение задачи с Лапласом
    const Types::VectorXd new_rhs = T_matrix * rhs;
    const Types::VectorXd res_laplace =
        (-2 * 3.1415926) * Research::solve_without_precond<Eigen::GMRES>(laplacian, new_rhs, 1000, 1e-2);

    // собираем поверхностное поле
    auto field = Math::SurfaceScalarField<Types::scalar>::fromSLAESolution(mesh, res);
    field.setName("Solution");
    auto field_laplace = Math::SurfaceScalarField<Types::scalar>::fromSLAESolution(mesh, res_laplace);
    field_laplace.setName("Solution_Laplace");

    auto field_new_rhs = Math::SurfaceScalarField<Types::scalar>::fromSLAESolution(mesh, T_matrix * res_laplace);
    field_new_rhs.setName("new_rhs");

    VTK::united_snapshot(Containers::vector{field, field_laplace, field_new_rhs}, {}, mesh, path_to_res + "paraview_snapshots/");
}
