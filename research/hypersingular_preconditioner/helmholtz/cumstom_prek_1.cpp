//
// Created by evgen on 04.09.2025.
//
#include "../OperatorT_simple.hpp"
#include "../OperatorS_simple.hpp"

#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "../matrix/MatrixPreconditioned.hpp"
#include "../matrix/MatrixTraits.hpp"

#include "visualisation/VTKFunctions.hpp"

#include "../Solve.hpp"
#include "../matrix/CustomPreconditioner.hpp"

#include "unsupported/Eigen/IterativeSolvers"

#include <string>

using namespace EMW;

using Preconditioner = Research::Matrix::Preconditioning::CustomPreconditioner<Types::complex_d, Types::MatrixXc>;
using SpecialMatrixType = Research::Matrix::Wrappers::MatrixReplacementComplex<Types::MatrixXc, Preconditioner>;

int main() {
    long unsigned int N_POINTS = 100;

    const std::string file_nodes =
        "/home/evgen/Education/Schools/Sirius2025/meshes/" + std::to_string(N_POINTS) + "/nodes.csv";
    const std::string file_cells =
        "/home/evgen/Education/Schools/Sirius2025/meshes/" + std::to_string(N_POINTS) + "/cells.csv";
    const std::string path_to_res = "/home/evgen/Education/MasterDegree/thesis/my_papers/Hypersingular_preconditioner/direct_preconditioning/";

    // делаем сетку
    const auto [nodes, cells, tag] = EMW::Parser::parse_mesh_without_tag(file_nodes, file_cells);
    auto mesh = EMW::Mesh::SurfaceMesh(nodes, cells);
    mesh.setName("Helmholtz_curtom_prec");
    const Types::index n_cells = mesh.getCells().size();

    // задаем волновое число
    const Types::complex_d k = {10, 0};

    // делаем поле правой части на сетке
    const auto rhs_field =
        Math::SurfaceScalarField<Types::complex_d>::fromSLAESolution(mesh, Types::VectorXc::Ones(n_cells));
    const Types::VectorXc rhs = rhs_field.formVector();

    // дискретизация гиперсингулярного интеграла по заданной сетке
    const Types::MatrixXc T_matrix = EMW::OperatorT::HelmholtzEquation::T_over_mesh(k, mesh, mesh);
    // собираем матрицу, обратную к предобуславливателю
    const Types::MatrixXc S_matrix = EMW::OperatorS::Helmholtz::S_over_mesh(k, mesh, mesh);

    const auto special_matrix = SpecialMatrixType{T_matrix, S_matrix, N_POINTS, 2. / (N_POINTS-1), k};

    // I. обычное решение уравнения Tg = f
    const auto res_ordinary = Research::solve_without_precond<Eigen::GMRES>(T_matrix, rhs, 1000, 1e-14);

    // II. решение уравнения с левым предобуславливателем:
    const auto res_precond = Research::solve<Eigen::GMRES>(special_matrix, rhs, 1000, 1e-14);

    // собираем поверхностное поле
    auto field = Math::SurfaceScalarField<Types::complex_d>::fromSLAESolution(mesh, res_ordinary);
    field.setName("Solution_ord");
    auto field_precond = Math::SurfaceScalarField<Types::complex_d>::fromSLAESolution(mesh, res_precond);
    field_precond.setName("Solution_precond");

    VTK::united_snapshot(Containers::vector{field, field_precond}, {}, mesh, path_to_res + "paraview_snapshots/");
}
