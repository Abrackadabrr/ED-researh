//
// Created by evgen on 06.08.2025.
//

#include "../OperatorT_simple.hpp"
#include "../DiscreteLaplacian.hpp"

#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"

#include "visualisation/VTKFunctions.hpp"

#include "../Solve.hpp"

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
    mesh.setName("Helmholtz");
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
    const auto rhs_field = Math::SurfaceScalarField<Types::complex_d>::fromSLAESolution(mesh, Types::VectorXc::Ones(n_cells));
    const auto rhs = rhs_field.formVector();

    const Types::complex_d k = {10., 0};

    // дискретизация гиперсингулярного интеграла по заданной сетке
    const auto T_matrix = EMW::OperatorT::HelmholtzEquation::T_over_mesh(k, mesh, mesh);

    // решение прямой задачи
    const auto res = Research::solve_without_precond<Eigen::GMRES>(T_matrix, rhs, 1000, 1e-8);

    // cбор дискретного оператора лапласа
    const auto helmholtz = DiscreteLaplacian::discreteHelmholtz(k, N_POINTS + 2, N_POINTS + 2, 2. /(N_POINTS - 1), 2. / (N_POINTS - 1));

    // решение задачи с Лапласом
    const Types::VectorXc support_rhs = T_matrix * rhs;
    const Types::VectorXc res_helm = (-2 * 3.1415926) * Research::solve_without_precond<Eigen::GMRES>(helmholtz, support_rhs, 1000, 1e-2);

    // собираем поверхностное поле
    auto field = Math::SurfaceScalarField<Types::complex_d>::fromSLAESolution(mesh, res);
    field.setName("Solution");
    auto field_laplace = Math::SurfaceScalarField<Types::complex_d>::fromSLAESolution(mesh, res_helm);
    field_laplace.setName("Solution Helmholtz");

    VTK::united_snapshot(Containers::vector{field, field_laplace}, {}, mesh, path_to_res + "paraview_snapshots/");
}
