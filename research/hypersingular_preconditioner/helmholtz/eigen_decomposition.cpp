//
// Created by evgen on 15.09.2025.
//

#include "operators/scalar/OperatorK.hpp"
#include "operators/scalar/OperatorS.hpp"
#include "operators/scalar/OperatorT.hpp"

#include "Utils.hpp"

#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"

#include "VTKFunctions.hpp"

#include <Eigen/Eigenvalues>

#include <string>

#include <fstream>

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

using namespace EMW;

template <typename seq_container_like_t>
void write_result(seq_container_like_t &&vec, const std::string &path, std::string postfix) {
    Containers::vector<int> indexes{};
    indexes.resize(vec.size());
    std::ranges::iota(indexes.begin(), indexes.end(), 0);
    // Запись сингулярных чисел
    std::ofstream file(path + "sv_" + postfix + ".csv");
    Utils::to_csv(indexes, vec, "ind", "sv", file, '\t');
}

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

template <typename Matrix> void dump(const Matrix &matrix, const std::string &filename) {
    std::ofstream file(filename);
    file.precision(20);
    for (Types::index row = 0; row < matrix.rows(); ++row) {
        for (Types::index col = 0; col < matrix.cols(); ++col) {
            file << matrix(row, col).real() << "+" << matrix(row, col).imag() << "j";
            if (col != matrix.cols() - 1)
                file << " ";
        }
        file << std::endl;
    }
    file.close();
}

#define PLATE 0

int main() {
#if PLATE
    long unsigned int N_POINTS = 20;
    const auto mesh = read_plate_mesh(N_POINTS);
#else
    // считываем сетку
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "cube/168_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "cube/144_cells.csv";
    // собираем сетки
    const auto parser_out = EMW::Parser::parseMesh(nodesFile, cellsFile);
    auto mesh = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells, parser_out.tags};

#endif
    const std::string path_to_res = "/home/evgen/Education/MasterDegree/thesis/my_papers/Hypersingular_preconditioner/";

    // рисуем сетку
    VTK::surface_snapshot(mesh, path_to_res);

    // задаем волновое число
    const Types::complex_d k = {10, 0};

    // дискретизация оператора T по заданной сетке
    const EMW::Operators::T_operator operator_T(k, 1e-6);
    const Types::MatrixXc T_matrix = operator_T.matrix(mesh, mesh);
    // дискретизация оператора S по заданной сетке
    const EMW::Operators::S_operator operator_S(k, 1e-6);
    const Types::MatrixXc S_matrix = operator_S.matrix(mesh, mesh);
    // дискретизация оператора К' по заданной сетке
    const Operators::K_operator operator_K(k);
    const Types::MatrixXc K_dash_matrix = operator_K.matrix(mesh, mesh);

    // 1) Считаем разложение Шура матрицы T
    Eigen::ComplexSchur<Types::MatrixXc> eigen_T;
    eigen_T.compute(T_matrix.conjugate());
    const Types::VectorXc eig_t_transpose = eigen_T.matrixT().diagonal();

    eigen_T.compute(T_matrix);
    const Types::VectorXc eig_t_straight = eigen_T.matrixT().diagonal();

    std::cout << "Разность между собственными чиcлами T и T^T: "
              << (eig_t_transpose - eig_t_straight.conjugate().reshaped(eig_t_transpose.size(), 1)).norm() /
                     eig_t_straight.norm()
              << std::endl;

    write_result(eig_t_straight, path_to_res + "preconditioning/spectral/", "sphere");
#if 1
    // 2) Считаем разложение Шура T * S;
    Eigen::ComplexSchur<Types::MatrixXc> eigen_ST;
    const Types::MatrixXc TS_matrix = T_matrix * S_matrix;

    eigen_ST.compute((TS_matrix).conjugate());
    const Types::VectorXc eig_vals_transpose = eigen_ST.matrixT().diagonal();

    eigen_ST.compute(TS_matrix);
    const Types::VectorXc eig_vals_straight = eigen_ST.matrixT().diagonal();

    std::cout << "Разность между собственными чиcлами TS и (TS)^T: "
              << (eig_vals_straight - eig_vals_transpose.conjugate().reshaped(eig_t_transpose.size(), 1)).norm() /
                     eig_vals_straight.norm()
              << std::endl;

    write_result(eig_vals_straight, path_to_res + "preconditioning/spectral/", "sphere_prec");
#endif
#if 1
    // 3) Считаем разложение Шура матрицы T * S * (I - K'^2)^{-1}
    const Types::MatrixXc inversed_term =
        (Types::MatrixXc::Identity(K_dash_matrix.rows(), K_dash_matrix.cols()) - (K_dash_matrix * K_dash_matrix))
            .inverse();
    Eigen::ComplexSchur<Types::MatrixXc> eigen_kolton;
    const Types::MatrixXc TSinv_matrix = TS_matrix * inversed_term;

    eigen_kolton.compute((TSinv_matrix).conjugate());
    const Types::VectorXc eig_vals_kolton_transpose = eigen_kolton.matrixT().diagonal();

    eigen_kolton.compute(TSinv_matrix);
    const Types::VectorXc eig_vals_kolton_straight = eigen_kolton.matrixT().diagonal();

    std::cout << "Разность между собственными чиcлами TS(I + K^2) и (TS(I + K^2))^T: "
              << (eig_vals_kolton_straight -
                  eig_vals_kolton_transpose.conjugate().reshaped(eig_vals_kolton_transpose.size(), 1)).norm() /
                     eig_vals_kolton_straight.norm()
              << std::endl;

    write_result(eig_vals_kolton_straight, path_to_res + "preconditioning/spectral/", "sphere_prec_kolton");
#endif
}
