//
// Created by evgen on 08.08.2025.
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
    long unsigned int N_POINTS = 10;
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

    // дискретизация оператора S по заданной сетке
    const EMW::Operators::S_operator operator_S(k, 1e-3);
    const Types::MatrixXc S_matrix = OperatorS::Helmholtz::S_over_mesh(k, mesh, mesh);

    std::cout << "det(S) = " << S_matrix.determinant() << std::endl;

    std::cout << "(S - S^T).norm() = " << (S_matrix - S_matrix.transpose()).norm() << std::endl;
}
