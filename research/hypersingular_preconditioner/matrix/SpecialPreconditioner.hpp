//
// Created by evgen on 07.08.2025.
//

#ifndef SPECIALPRECONDITIONER_HPP
#define SPECIALPRECONDITIONER_HPP

#include "MatrixTraits.hpp"

#include "types/Types.hpp"

#include "Eigen/SparseCore"

#include <iostream>

namespace Research::Matrix::Preconditioning {

using namespace EMW;

template <typename scalar_t, typename matrix_t> class LaplacianPreconditioner {
    using vector_t = Types::VectorX<scalar_t>;
    // геометрия
    Types::index N, M;
    Types::scalar h1, h2;
    // параметры метода
    Types::scalar tolerance = 1e-4;
    Types::index max_iterations = 200;
    // матрицы для слау
    Eigen::SparseMatrix<Types::scalar> laplacian;
    const matrix_t &T; // очень опасно, но мы предполагаем, что матрица лежит в памяти
    Eigen::SparseMatrix<Types::scalar> projector;
    Eigen::SparseMatrix<Types::scalar> inverse_projector;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Types::scalar>> method;

  public:
    /**
     * Предобуславливатель строится по геометрии (N, M, h1, h2) и по исходной
     * матрице системы (она участвует в решении дополнительной системы)
     */
    LaplacianPreconditioner(const matrix_t &matrix, Types::index N_, Types::index M_, Types::scalar h1_,
                            Types::scalar h2_)
        : N(N_), M(M_), h1(h1_), h2(h2_), T(matrix), laplacian(DiscreteLaplacian::discreteLaplacian(N, M, h1, h2)),
          inverse_projector(DiscreteLaplacian::getInverseProjector(N, M)),
          projector(DiscreteLaplacian::getProjector(N, M)) {
        method.compute(laplacian);
        method.setMaxIterations(max_iterations);
        method.setTolerance(tolerance);
        std::cout << "Special preconditioner for equation is constructed" << std::endl;
    };

    // Решается система уравнений Lap * x = P * T * rsh,
    // затем решение поднимается в нормальное пространство y = inv_P * x
    vector_t solve(const vector_t &rhs) const {
        const vector_t new_rsh = projector * (T * rhs);
        const vector_t res = method.solve(new_rsh);
        std::cout << method.iterations() << std::endl;
        std::cout << res.norm() << std::endl;
        if (method.info() != Eigen::Success) {
            throw std::runtime_error("Laplacian preconditioner failed");
        }
        return inverse_projector * res;
    }
};

template <typename scalar_t, typename matrix_t> class LaplacianStupidPreconditioner {
    using vector_t = Types::VectorX<scalar_t>;
    // геометрия
    Types::index N, M;
    Types::scalar h1, h2;
    // матрицы для образования прекоднера
    Eigen::SparseMatrix<Types::scalar> laplacian;
    const matrix_t &T; // очень опасно, но мы предполагаем, что матрица лежит в памяти
    Eigen::SparseMatrix<Types::scalar> projector;
    Eigen::SparseMatrix<Types::scalar> inverse_projector;
    Eigen::MatrixXd preconditioner;

  public:
    /**
     * Предобуславливатель строится по геометрии (N, M, h1, h2) и по исходной
     * матрице системы (она участвует в решении дополнительной системы)
     */
    LaplacianStupidPreconditioner(const matrix_t &matrix, Types::index N_, Types::index M_, Types::scalar h1_,
                                  Types::scalar h2_)
        : N(N_), M(M_), h1(h1_), h2(h2_), T(matrix), laplacian(DiscreteLaplacian::discreteLaplacian(N, M, h1, h2)),
          inverse_projector(DiscreteLaplacian::getInverseProjector(N, M)),
          projector(DiscreteLaplacian::getProjector(N, M)) {
        preconditioner = inverse_projector * Types::MatrixXd(laplacian).inverse() * projector * T;
        std::cout << "Special preconditioner for equation is constructed" << std::endl;
    };

    // Решается система уравнений Lap * (P * x) = P * T * rsh,
    // а мы имеем прямо-таки обратную матрицу, так что x = (P_i * (Lap)^{-1} * P * T) * rsh
    vector_t solve(const vector_t &rhs) const { return preconditioner * (rhs); }
};

template <typename scalar_t, typename matrix_t> class LeftLaplacianPreconditioner {
    using vector_t = Types::VectorX<scalar_t>;
    // геометрия
    Types::index N, M;
    Types::scalar h1, h2;
    // параметры метода
    Types::scalar tolerance = 1e-3;
    Types::index max_iterations = 1000;
    // матрицы для слау
    Eigen::SparseMatrix<Types::scalar> extended_laplacian;
    const matrix_t &T; // очень опасно, но мы предполагаем, что матрица лежит в памяти
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Types::scalar>> method;

  public:
    /**
     * Предобуславливатель строится по геометрии (N, M, h1, h2) и по исходной
     * матрице системы (она участвует в решении дополнительной системы)
     */
    LeftLaplacianPreconditioner(const matrix_t &matrix, Types::index N_, Types::index M_, Types::scalar h1_,
                                Types::scalar h2_)
        : N(N_), M(M_), h1(h1_), h2(h2_), T(matrix),
          extended_laplacian(DiscreteLaplacian::discreteLaplacian(N + 2, M + 2, h1, h2)) {
        method.compute(extended_laplacian);
        method.setMaxIterations(max_iterations);
        method.setTolerance(tolerance);
        std::cout << "Special preconditioner for equation is constructed" << std::endl;
    };

    // Решается система уравнений Lap * x = T * rsh,
    vector_t solve(const vector_t &rhs) const {
        const vector_t new_rsh = T * rhs;
        const vector_t res = method.solve(new_rsh);
        if (method.info() != Eigen::Success) {
            throw std::runtime_error("Laplacian preconditioner failed");
        }
        return res;
    }
};

template <typename scalar_t, typename matrix_t> class HelmholtzPreconditioner {
    using vector_t = Types::VectorX<scalar_t>;
    // геометрия
    Types::index N, M;
    Types::scalar h1, h2;
    // параметры метода
    Types::scalar tolerance = 1e-1;
    Types::index max_iterations = 5000;
    // матрицы для слау
    Eigen::SparseMatrix<Types::complex_d> helm;
    const matrix_t &T; // очень опасно, но мы предполагаем, что матрица лежит в памяти
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Types::complex_d>> method;

  public:
    /**
     * Предобуславливатель строится по геометрии (N, M, h1, h2) и по исходной
     * матрице системы (она участвует в решении дополнительной системы)
     */
    HelmholtzPreconditioner(const matrix_t &matrix, Types::complex_d k, Types::index N_, Types::index M_,
                            Types::scalar h1_, Types::scalar h2_)
        : N(N_), M(M_), h1(h1_), h2(h2_), T(matrix),
          helm(DiscreteLaplacian::discreteHelmholtz(k, N + 2, M + 2, h1, h2)) {
        method.compute(helm);
        method.setMaxIterations(max_iterations);
        method.setTolerance(tolerance);
        std::cout << "Special preconditioner for equation is constructed" << std::endl;
    };

    // Решается система уравнений Lap * x = T * rsh,
    vector_t solve(const vector_t &rhs) const {
        const vector_t new_rsh = T * rhs;
        const vector_t res = method.solve(new_rsh);
        if (method.info() != Eigen::Success) {
            throw std::runtime_error("Helm preconditioner failed");
        }
        return res;
    }
};

template <typename scalar_t, typename matrix_t> class LaplacianComplexPreconditioner {
    using vector_t = Types::VectorX<scalar_t>;
    // геометрия
    Types::index N, M;
    Types::scalar h1, h2;
    // параметры метода
    Types::scalar tolerance = 1e-2;
    Types::index max_iterations = 400;
    // матрицы для слау
    Eigen::SparseMatrix<Types::complex_d> extended_laplacian;
    const matrix_t &T; // очень опасно, но мы предполагаем, что матрица лежит в памяти
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Types::complex_d>> method;

  public:
    /**
     * Предобуславливатель строится по геометрии (N, M, h1, h2) и по исходной
     * матрице системы (она участвует в решении дополнительной системы)
     */
    LaplacianComplexPreconditioner(const matrix_t &matrix, Types::index N_, Types::index M_, Types::scalar h1_, Types::scalar h2_)
        : N(N_), M(M_), h1(h1_), h2(h2_), T(matrix), extended_laplacian(DiscreteLaplacian::discreteComplexLaplacian(N + 2, M + 2, h1, h2)) {
        method.compute(extended_laplacian);
        method.setMaxIterations(max_iterations);
        method.setTolerance(tolerance);
        std::cout << "Special preconditioner for equation is constructed" << std::endl;
    };

    // Решается система уравнений Lap * x = T * rsh,
    vector_t solve(const vector_t &rhs) const {
        const vector_t new_rsh = T * rhs;
        const vector_t res = method.solve(new_rsh);
        if (method.info() != Eigen::Success) {
            throw std::runtime_error("Laplacian preconditioner failed");
        }
        return res;
    }
};



} // namespace EMW::Math::LinAgl::Matrix::Preconditioning


#endif //SPECIALPRECONDITIONER_HPP
