//
// Created by evgen on 03.10.2025.
//
#include "operators/scalar/OperatorK.hpp"
#include "operators/scalar/OperatorS.hpp"
#include "operators/scalar/OperatorT.hpp"
#include "../OperatorT_simple.hpp"
#include "../OperatorS_simple.hpp"

#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "../matrix/MatrixPreconditioned.hpp"
#include "../matrix/MatrixTraits.hpp"

#include "visualisation/VTKFunctions.hpp"

#include "../Solve.hpp"
#include "../matrix/DirectSolutionPreconditioner.hpp"

#include "unsupported/Eigen/IterativeSolvers"

#include <string>

using namespace EMW;

using Preconditioner = Research::Matrix::Preconditioning::DirectPreconditioner<Types::complex_d, Types::MatrixXc>;
using SpecialMatrixType = Research::Matrix::Wrappers::MatrixReplacementComplex<Types::MatrixXc, Preconditioner>;

Mesh::SurfaceMesh read_plate_mesh(Types::index N_POINTS) {
    const std::string file_nodes =
        "/home/evgen/Education/Schools/Sirius2025/meshes/" + std::to_string(N_POINTS) + "/nodes.csv";
    const std::string file_cells =
        "/home/evgen/Education/Schools/Sirius2025/meshes/" + std::to_string(N_POINTS) + "/cells.csv";

    // делаем сетку
    const auto [nodes, cells, tag] = EMW::Parser::parse_mesh_without_tag(file_nodes, file_cells);
    auto mesh = EMW::Mesh::SurfaceMesh(nodes, cells);
    mesh.setName("Helmholtz_curtom_prec");
    return mesh;
}

#define PLATE 1

int main() {
#if PLATE
    long unsigned int N_POINTS = 50;
    const auto mesh = read_plate_mesh(N_POINTS);
#else
    // считываем сетку
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "cube/840_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "cube/784_cells.csv";
    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile);
    auto mesh = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells, parser_out.tags};

#endif
    // задаем волновое число
    const Types::complex_d k = {10, 0};

    // делаем поле правой части на сетке
    const auto rhs_field = Math::SurfaceScalarField<Types::complex_d>::fromSLAESolution(
        mesh, Types::VectorXc::Ones(mesh.getCells().size()));
    const Types::VectorXc rhs = rhs_field.formVector();

    // дискретизация оператора T по заданной сетке
    const Operators::T_operator_on_plane operator_T(k, 1e-3);
    const Types::MatrixXc T_matrix = operator_T.matrix(mesh, mesh);
    const Operators::T_operator new_operator_T(k, 1e-3);
    const Types::MatrixXc new_T_matrix = new_operator_T.matrix(mesh, mesh);

    std::cout << (new_T_matrix.transpose() - new_T_matrix).norm() / new_T_matrix.norm() << std::endl;

    // дискретизация оператора S по заданной сетке
    const EMW::Operators::S_operator operator_S(k, 1e-3);
    const Types::MatrixXc S_matrix = OperatorS::Helmholtz::S_over_mesh(k, mesh, mesh);

    // собираем матрицы с двумя разными предобуславливателями
    const auto special_matrix_1 = SpecialMatrixType{T_matrix, S_matrix};

    std::cout << "det(S) = " << S_matrix.determinant() << std::endl;

    // I. обычное решение уравнения Tg = f
    const auto res_ordinary = Research::solve_without_precond<Eigen::GMRES>(T_matrix, rhs, 1000, 1e-14);

    // II. решение уравнения с левым предобуславливателем: T (S g) = f (пердобуславилватель А = S^{-1})
    const auto res_precond = Research::solve<Eigen::GMRES>(special_matrix_1, rhs, 1000, 1e-14);

    // Сравнение решений
    const Types::scalar err1 = (res_ordinary - res_precond).norm();
    const Types::scalar rel_err1 = (res_ordinary - res_precond).norm() / res_ordinary.norm();

    std::cout << "Абсолютная онрма ошибки: " << err1 << "; Относительная норма ошибки: " << rel_err1 << std::endl;
    std::cout << "Норма прямого решения: " << res_ordinary.norm() << "; Норма кривого решения: " << res_precond.norm();
}
