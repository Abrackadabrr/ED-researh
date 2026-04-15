//
// Created by evgen on 07.08.2025.
//

#include "DiscreteLaplacian.hpp"
#include "OperatorT_simple.hpp"
#include "OperatorS_simple.hpp"

#include "iterative_solvers/SimpleIterationMethod.hpp"

#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"

#include "matrix/MatrixPreconditioned.hpp"
#include "matrix/MatrixTraits.hpp"

#include "visualisation/VTKFunctions.hpp"

#include "Solve.hpp"
#include "matrix/SpecialPreconditioner.hpp"

#include "unsupported/Eigen/IterativeSolvers"

#include <string>

using namespace EMW;

using Preconditioner = Research::Matrix::Preconditioning::LeftLaplacianPreconditioner<Types::scalar, Types::MatrixXd>;
using SpecialMatrixType = Research::Matrix::Wrappers::MatrixReplacementReal<Types::MatrixXd, Preconditioner>;

int main() {
    long unsigned int N_POINTS = 100;

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
    const Types::VectorXd rhs = rhs_field.formVector();

    // дискретизация гиперсингулярного интеграла по заданной сетке
    const Types::MatrixXd T_matrix = EMW::OperatorT::LaplaceEquation::T_over_mesh(mesh, mesh);

    const auto special_matrix =
        SpecialMatrixType{T_matrix, N_POINTS, N_POINTS, 2. / (N_POINTS - 1), 2. / (N_POINTS - 1)};

    // I. обычное решение уравнения Tg = f
    const auto res_ordinary = Research::solve_without_precond<Eigen::GMRES>(T_matrix, rhs, 1000, 1e-14);
    Eigen::GMRES<Types::MatrixXd> method{};

    // II. решение уравнения с левым предобуславливателем: T P_i g = f
    // 1) найти новую правую часть
    // 2) решить систему с изменённой матрицей
    const auto res_precond = Research::solve<Eigen::GMRES>(special_matrix, rhs, 1000, 1e-14);

    // собираем поверхностное поле
    auto field = Math::SurfaceScalarField<Types::scalar>::fromSLAESolution(mesh, res_ordinary);
    field.setName("Solution_ord");
    auto field_precond = Math::SurfaceScalarField<Types::scalar>::fromSLAESolution(mesh, res_precond);
    field_precond.setName("Solution_precond");

    VTK::united_snapshot(Containers::vector{field, field_precond}, {}, mesh, path_to_res + "paraview_snapshots/");
}
