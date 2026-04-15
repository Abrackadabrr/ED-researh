//
// Created by evgen on 18.11.2024.
//

#include "examples/pathes.hpp"
#include "experiment/PhysicalCondition.hpp"
#include "math/MathConstants.hpp"
#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"
#include "mesh/Algorithms.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"
#include "mesh/Utils.hpp"
#include "meshes/plate/PlateGrid.hpp"
#include "operators/Operators.hpp"
#include "slae_generation/MatrixGeneration.hpp"
#include "visualisation/VTKFunctions.hpp"

#include "iterative_solvers/SimpleIterationMethod.hpp"
#include "slae_generation/Preconditioning.hpp"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SVD>
#include <unsupported/Eigen/IterativeSolvers>

#include <chrono>
#include <filesystem>
#include <iostream>

using namespace EMW;
using namespace EMW::Types;

Types::Vector3c getConstantRHS(const Types::Vector3d &x) {
    if (x.norm() < 1)
        return {complex_d{1, 0}, {0, 0}, {0, 0}};
    return Vector3c::Zero();
}

std::pair<VectorXc, int> solve(const MatrixXc &AP, const MatrixXc &P, const VectorXc &b, scalar tolerance) {
    auto method = Eigen::GMRES<MatrixXc, Eigen::IdentityPreconditioner>{};

    method.setMaxIterations(30000);
    std::cout << method.maxIterations() << std::endl;
    method.setTolerance(tolerance);
    method.set_restart(30000);

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    method.compute(AP);
    VectorXc j_vec = VectorXc{P * method.solveWithGuess(b, b)};

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::ratio<1, 1>> elapsed_seconds = end - start;
    std::cout << "Время решения GMRES: " << elapsed_seconds.count() << std::endl;
    std::cout << "total iterations: " << method.iterations() << std::endl;
    std::cout << "tolerance: " << method.tolerance() << std::endl;
    std::cout << "total error: " << method.error() << std::endl;
    std::cout << "Info: " << static_cast<int>(method.info()) << std::endl;
    return {j_vec, method.iterations()};
}

const std::string path = Pathes::studies + "plane/preconditioning/rectangular/convergence/preconditioned";

void solveWithSteps(const Math::SurfaceVectorField &rhsField, const MatrixXc &AP, const MatrixXc &P, const VectorXc &b,
                    const Types::complex_d k, scalar tolerance, const Types::index total_iterations) {
    std::vector<Types::VectorXc> steps, resuduals;
    steps.resize(total_iterations);
    resuduals.resize(total_iterations);
    const Mesh::SurfaceMesh &surfaceMesh = rhsField.getManifold();

#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < total_iterations; ++i) {
        auto method = Eigen::GMRES<MatrixXc, Eigen::IdentityPreconditioner>{};
        method.setTolerance(tolerance);
        method.compute(AP);
        std::cout << i << std::endl;
        method.setMaxIterations(i);
        method.set_restart(i);
        const VectorXc solution = method.solveWithGuess(b, b);
        steps[i - 1] = P * solution;
        resuduals[i - 1] = b - AP * solution;
    }

    auto method = Eigen::GMRES<MatrixXc, Eigen::IdentityPreconditioner>{};
    method.setTolerance(tolerance);
    method.compute(AP);
    std::cout << total_iterations << std::endl;
    method.setMaxIterations(total_iterations);
    method.set_restart(total_iterations);
    const VectorXc solution = method.solveWithGuess(b, b);
    steps[total_iterations - 1] = P * solution;
    resuduals[total_iterations - 1] = b - AP * solution;

    std::cout << "total iterations: " << method.iterations() << std::endl;
    std::cout << "tolerance: " << method.tolerance() << std::endl;
    std::cout << "total error: " << method.error() << std::endl;
    std::cout << "Info: " << static_cast<int>(method.info()) << std::endl;

    std::for_each(steps.begin(), steps.end(), [&steps](const VectorXc &x) -> VectorXc { return x - steps.back(); });
    for (int i = 1; i < total_iterations; ++i) {
        Math::SurfaceVectorField solField = Math::SurfaceVectorField::TangentField(surfaceMesh, steps[i - 1]);
        solField.setName("solution");
        Math::SurfaceVectorField diffField =
            Math::SurfaceVectorField::TangentField(surfaceMesh, steps[i - 1] - steps.back());
        diffField.setName("difference");
        Math::SurfaceVectorField resField = Math::SurfaceVectorField::TangentField(surfaceMesh, resuduals[i - 1]);
        resField.setName("residual");
        VTK::united_snapshot({solField, diffField, resField, rhsField}, {}, surfaceMesh, path + "/", i);
        std::cout << i << " dumped" << std::endl;
    }
}

int main() {
    Eigen::setNbThreads(14);

    // std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);

    int N1 = 61;
    scalar h1 = 1. / (N1 - 1);
    int N2 = 61;
    scalar h2 = 1. / (N2 - 1);
    auto surfaceMesh = Examples::Plate::generateRectangularMesh(N1, N2, h1, h2);

    surfaceMesh.setName("rect" + std::to_string(N1) + "_x_" + std::to_string(N2));

    // падающее поле на треугольной сетке
    constexpr complex_d k = 4 * Math::Constants::PI<scalar>() * complex_d{1, 0.05};
    const Vector3d e = Vector3d{1, 0, 0}.normalized();
    EMW::Physics::planeWaveCase physics{e, k, Vector3d{0, 0, -1}.normalized()};
    const auto initial_field_function = [physics](const Mesh::point_t &point) -> Vector3c {
        // std::cout << physics.value(point) << std::endl;
        return physics.value(point);
    };

    const Types::scalar tolerance = 1e-2;

    auto rhsField = Math::SurfaceVectorField(surfaceMesh, getConstantRHS);
    const auto A = Matrix::getMatrix(k, surfaceMesh);
    const auto P = Matrix::Preconditioning::getPreconditiotner(surfaceMesh, 2, k);
    // MatrixXc::Identity(2 * surfaceMesh.getCells().size(), 2 * surfaceMesh.getCells().size())
    const MatrixXc AP = A * P;

    const auto solution = solve(AP, P, rhsField.asSLAERHS(), tolerance);

    solveWithSteps(rhsField, AP, P, rhsField.asSLAERHS(), k, 1e-2,  solution.second);
}
