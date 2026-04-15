//
// Created by evgen on 07.08.2025.
//

#ifndef DISCRETELAPLACIAN_HPP
#define DISCRETELAPLACIAN_HPP

#include <Eigen/SparseCore>

#include <iostream>

#include "types/Types.hpp"

using namespace EMW;

namespace DiscreteLaplacian {
// вспомогательные функции
namespace detail {
Containers::set<Types::index> get_boundary_cells_indexes(Types::index N) {
    N = N - 1;
    const Types::index N_CELLS = 4 * N - 4;
    Containers::set<Types::index> boundary_cells_indexes;
    for (Types::index i = 0; i < N; ++i) {
        boundary_cells_indexes.insert(i);
    }
    for (Types::index i = 0; i < N - 2; ++i) {
        const Types::index i_1 = (i + 1) * N;
        const Types::index i_2 = (i + 1) * N + (N - 1);
        boundary_cells_indexes.insert(i_1);
        boundary_cells_indexes.insert(i_2);
    }
    for (Types::index i = 0; i < N; ++i) {
        boundary_cells_indexes.insert(i + N * (N - 1));
    }
    if (boundary_cells_indexes.size() != N_CELLS) {
        std::cout << boundary_cells_indexes.size() << std::endl;
        std::cout << N_CELLS << std::endl;
        throw std::runtime_error("Buuuuuuuuuuuuuuuug");
    }
    return boundary_cells_indexes;
}
} // namespace detail

Eigen::SparseMatrix<Types::scalar> discreteLaplacian(Types::index N, Types::index M, Types::scalar h1,
                                                     Types::scalar h2) {
    N = N - 1;
    M = M - 1;
    Types::index n_rows = (N - 2) * (M - 2);
    Types::index n_cols = n_rows;
    Types::index interior_points = n_rows;

    Eigen::SparseMatrix<Types::scalar> laplacian(interior_points, interior_points);

    double inv_h1_sq = 1.0 / (h1 * h1);
    double inv_h2_sq = 1.0 / (h2 * h2);
    double diag_val = -2.0 * (inv_h1_sq + inv_h2_sq);

    // Reserve space for 5-point stencil (4 neighbors + diagonal)
    laplacian.reserve(Eigen::VectorXi::Constant(interior_points, 5));

    for (int j = 0; j < M - 2; ++j) {
        for (int i = 0; i < N - 2; ++i) {
            int index = i + j * (N - 2);

            // Diagonal entry
            laplacian.insert(index, index) = diag_val;

            // x-direction neighbors (Dirichlet BCs are implicit)
            if (i > 0)
                laplacian.insert(index, index - 1) = inv_h1_sq; // left
            if (i < N - 3)
                laplacian.insert(index, index + 1) = inv_h1_sq; // right

            // y-direction neighbors
            if (j > 0)
                laplacian.insert(index, index - (N - 2)) = inv_h2_sq; // bottom
            if (j < M - 3)
                laplacian.insert(index, index + (N - 2)) = inv_h2_sq; // top
        }
    }
    return laplacian;
}

Eigen::SparseMatrix<Types::complex_d> discreteComplexLaplacian(Types::index N, Types::index M, Types::scalar h1,
                                                     Types::scalar h2) {
    N = N - 1;
    M = M - 1;
    Types::index n_rows = (N - 2) * (M - 2);
    Types::index n_cols = n_rows;
    Types::index interior_points = n_rows;

    Eigen::SparseMatrix<Types::complex_d> laplacian(interior_points, interior_points);

    double inv_h1_sq = 1.0 / (h1 * h1);
    double inv_h2_sq = 1.0 / (h2 * h2);
    double diag_val = -2.0 * (inv_h1_sq + inv_h2_sq);

    // Reserve space for 5-point stencil (4 neighbors + diagonal)
    laplacian.reserve(Eigen::VectorXi::Constant(interior_points, 5));

    for (int j = 0; j < M - 2; ++j) {
        for (int i = 0; i < N - 2; ++i) {
            int index = i + j * (N - 2);

            // Diagonal entry
            laplacian.insert(index, index) = Types::complex_d{diag_val, 0};

            // x-direction neighbors (Dirichlet BCs are implicit)
            if (i > 0)
                laplacian.insert(index, index - 1) =  Types::complex_d{inv_h1_sq, 0}; // left
            if (i < N - 3)
                laplacian.insert(index, index + 1) =  Types::complex_d{inv_h1_sq, 0}; // right

            // y-direction neighbors
            if (j > 0)
                laplacian.insert(index, index - (N - 2)) =  Types::complex_d{inv_h2_sq, 0}; // bottom
            if (j < M - 3)
                laplacian.insert(index, index + (N - 2)) =  Types::complex_d{inv_h2_sq, 0}; // top
        }
    }
    return laplacian;
}

Eigen::SparseMatrix<Types::complex_d> discreteHelmholtz(Types::complex_d k, Types::index N, Types::index M,
                                                        Types::scalar h1, Types::scalar h2) {
    N = N - 1;
    M = M - 1;
    Types::index n_rows = (N - 2) * (M - 2);
    Types::index n_cols = n_rows;
    Types::index interior_points = n_rows;

    Eigen::SparseMatrix<Types::complex_d> helmholtz(interior_points, interior_points);

    double inv_h1_sq = 1.0 / (h1 * h1);
    double inv_h2_sq = 1.0 / (h2 * h2);
    double diag_val = -2.0 * (inv_h1_sq + inv_h2_sq);

    // Reserve space for 5-point stencil (4 neighbors + diagonal)
    helmholtz.reserve(Eigen::VectorXi::Constant(interior_points, 5));

    for (int j = 0; j < M - 2; ++j) {
        for (int i = 0; i < N - 2; ++i) {
            int index = i + j * (N - 2);

            // Diagonal entry
            helmholtz.insert(index, index) = diag_val + (k * k); // отличие от лапласа только здесь

            // x-direction neighbors (Dirichlet BCs are implicit)
            if (i > 0)
                helmholtz.insert(index, index - 1) = inv_h1_sq; // left
            if (i < N - 3)
                helmholtz.insert(index, index + 1) = inv_h1_sq; // right

            // y-direction neighbors
            if (j > 0)
                helmholtz.insert(index, index - (N - 2)) = inv_h2_sq; // bottom
            if (j < M - 3)
                helmholtz.insert(index, index + (N - 2)) = inv_h2_sq; // top
        }
    }
    return helmholtz;
}

Eigen::SparseMatrix<Types::scalar> getProjector(Types::index N, Types::index M) {
    N = N - 1;
    M = M - 1;
    Types::index n_points_inner = (N - 2) * (M - 2);
    Types::index n_points_outer = N * M;
    const auto boundary_element_indexes = detail::get_boundary_cells_indexes(N + 1);

    Eigen::SparseMatrix<Types::scalar> res(n_points_inner, n_points_outer);

    int row = 0;
    for (int i = 0; i < n_points_outer; ++i) {
        if (!boundary_element_indexes.contains(i)) {
            res.insert(row, i) = 1.0;
            row++;
        }
    }
    return res;
}

Eigen::SparseMatrix<Types::scalar> getInverseProjector(Types::index N, Types::index M) {
    N = N - 1;
    M = M - 1;
    Types::index n_points_inner = (N - 2) * (M - 2);
    Types::index n_points_outer = N * M;
    const auto boundary_element_indexes = detail::get_boundary_cells_indexes(N + 1);

    Eigen::SparseMatrix<Types::scalar> res(n_points_outer, n_points_inner);

    int col = 0;
    for (int i = 0; i < n_points_outer; ++i) {
        if (!boundary_element_indexes.contains(i)) {
            res.insert(i, col) = 1.0;
            col++;
        }
    }
    return res;
}

Eigen::SparseMatrix<Types::scalar> getBoundaryAnnulator(Types::index N, Types::index M) {
    N = N - 1;
    M = M - 1;
    Types::index n_points_outer = N * M;
    const auto boundary_element_indexes = detail::get_boundary_cells_indexes(N + 1);

    Eigen::SparseMatrix<Types::scalar> res(n_points_outer, n_points_outer);

    for (int i = 0; i < n_points_outer; ++i) {
        if (!boundary_element_indexes.contains(i)) {
            res.insert(i, i) = 1.0;
        }
    }
    return res;
}

#if 0
Eigen::SparseMatrix<Types::scalar> discreteExtendedLaplacian(Types::index N, Types::index M, Types::scalar h1,
                                                             Types::scalar h2) {
    N = N - 1;
    M = M - 1;
    Types::index n_rows = N * M;
    Types::index n_cols = n_rows;

    Eigen::SparseMatrix<Types::scalar> laplacian(n_rows, n_cols);
    const auto boundary_element_indexes = detail::get_boundary_cells_indexes(N + 1);

    double inv_h1_sq = 1.0 / (h1 * h1);
    double inv_h2_sq = 1.0 / (h2 * h2);
    double diag_val = -2.0 * (inv_h1_sq + inv_h2_sq);

    // Reserve space for 5-point stencil (4 neighbors + diagonal)
    laplacian.reserve(Eigen::VectorXi::Constant(n_rows, 5));

    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < N; ++i) {
            int index = i + j * N;
            if (!boundary_element_indexes.contains(index)) {
                // Diagonal entry
                laplacian.insert(index, index) = diag_val;

                // x-direction neighbors (Dirichlet BCs are implicit)
                if (i > 0)
                    laplacian.insert(index, index - 1) = inv_h1_sq; // left
                if (i < N - 1)
                    laplacian.insert(index, index + 1) = inv_h1_sq; // right

                // y-direction neighbors
                if (j > 0)
                    laplacian.insert(index, index - N) = inv_h2_sq; // bottom
                if (j < M - 1)
                    laplacian.insert(index, index + N) = inv_h2_sq; // top
            } else {
                laplacian.insert(index, index) = 1.;
            }
        }
    }
    return laplacian;
}
#endif

}

#endif //DISCRETELAPLACIAN_HPP
