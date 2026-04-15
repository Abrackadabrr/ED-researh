//
// Created by evgen on 03.03.2025.
//

#ifndef SOLVE_HPP
#define SOLVE_HPP

#include <chrono>
#include <iostream>

#include <unsupported/Eigen/IterativeSolvers>

#include "math/matrix/iterative_solvers_coverage/MatrixReplacement.hpp"
#include "types/Types.hpp"

namespace Research {

template <template <typename, typename> typename method_t, typename matrix_t, typename vector_t>
vector_t solve(const matrix_t &A, const vector_t &b, EMW::Types::index max_iterations, EMW::Types::scalar tolerance) {
    auto method = method_t<matrix_t, Eigen::IdentityPreconditioner>{};
    std::cout << "GMRES without precond" << std::endl;
    method.setMaxIterations(max_iterations);
    std::cout << method.maxIterations() << std::endl;
    method.setTolerance(tolerance);
    method.set_restart(max_iterations);

    auto start = std::chrono::steady_clock::now();

    method.compute(A);
    auto j_vec = vector_t{method.solve(b)};

    auto end = std::chrono::steady_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Время решения GMRES: " << elapsed_seconds.count() << std::endl;
    std::cout << "total iterations: " << method.iterations() << std::endl;
    std::cout << "tolerance: " << method.tolerance() << std::endl;
    std::cout << "total error: " << method.error() << std::endl;
    std::cout << "Info: " << static_cast<int>(method.info()) << std::endl;
    return j_vec;
}

template <template <typename> typename method_t, typename vector_t>
vector_t solve(const EMW::Types::MatrixXc &A, const vector_t &b, EMW::Types::index max_iterations,
               EMW::Types::scalar tolerance) {
    auto method = method_t<EMW::Types::MatrixXc>{};
    std::cout << "GMRES with native diagonal preconditioner" << std::endl;
    method.setMaxIterations(max_iterations);
    std::cout << "Max iterations: " << method.maxIterations() << std::endl;
    method.setTolerance(tolerance);
    std::cout << "Tolerance: " << tolerance << std::endl;
    method.set_restart(max_iterations);
    std::cout << "Restarts every: " << method.get_restart() << std::endl;

    auto start = std::chrono::steady_clock::now();

    method.compute(A);
    auto j_vec = vector_t{method.solve(b)};

    auto end = std::chrono::steady_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Время решения GMRES: " << elapsed_seconds.count() << std::endl;
    std::cout << "total iterations: " << method.iterations() << std::endl;
    std::cout << "tolerance: " << method.tolerance() << std::endl;
    std::cout << "total error: " << method.error() << std::endl;
    std::cout << "Info: " << static_cast<int>(method.info()) << std::endl;
    return j_vec;
}

template <template <typename, typename> typename method_t, typename vector_t>
vector_t solve_without_prec(const EMW::Types::MatrixXc &A, const vector_t &b, EMW::Types::index max_iterations,
               EMW::Types::scalar tolerance) {
  auto method = method_t<EMW::Types::MatrixXc, Eigen::IdentityPreconditioner>{};
  std::cout << "GMRES without prec" << std::endl;
  method.setMaxIterations(max_iterations);
  std::cout << "Max iterations: " << method.maxIterations() << std::endl;
  method.setTolerance(tolerance);
  std::cout << "Tolerance: " << tolerance << std::endl;
  method.set_restart(max_iterations);
  std::cout << "Restarts every: " << method.get_restart() << std::endl;

  auto start = std::chrono::steady_clock::now();

  method.compute(A);
  auto j_vec = vector_t{method.solve(b)};

  auto end = std::chrono::steady_clock::now();
  auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Время решения GMRES: " << elapsed_seconds.count() << std::endl;
  std::cout << "total iterations: " << method.iterations() << std::endl;
  std::cout << "tolerance: " << method.tolerance() << std::endl;
  std::cout << "total error: " << method.error() << std::endl;
  std::cout << "Info: " << static_cast<int>(method.info()) << std::endl;
  return j_vec;
}

template <template <typename, typename> typename method_t, typename matrix_t, typename precond_t, typename vector_t>
vector_t solve(const EMW::Math::LinAgl::Matrix::Wrappers::MatrixReplacement<matrix_t, precond_t> &A, const vector_t &b,
               EMW::Types::index max_iterations, EMW::Types::scalar tolerance) {
    using matrix_wrapper_t = EMW::Math::LinAgl::Matrix::Wrappers::MatrixReplacement<matrix_t, precond_t>;
    auto method = method_t<matrix_wrapper_t, Eigen::IdentityPreconditioner>{};

    method.setMaxIterations(max_iterations);
    std::cout << "Max GMRES iterations: " << method.maxIterations() << std::endl;
    method.setTolerance(tolerance);
    method.set_restart(500);

    std::cout << "Starting solution with 2ToepMatrix" << std::endl;

    auto start = std::chrono::steady_clock::now();

    method.compute(A);

    std::cout << "Matrix computed" << std::endl;

    auto j_vec = A.get_preconditioner().solve(vector_t{method.solve(b)});

    auto end = std::chrono::steady_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Время решения GMRES: " << elapsed_seconds.count() << std::endl;
    std::cout << "total iterations: " << method.iterations() << std::endl;
    std::cout << "tolerance: " << method.tolerance() << std::endl;
    std::cout << "total error: " << method.error() << std::endl;
    std::cout << "Info: " << static_cast<int>(method.info()) << std::endl;
    return j_vec;
}

}

#endif //SOLVE_HPP
