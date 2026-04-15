//
// Created by evgen on 11.11.2024.
//
#include "examples/pathes.hpp"
#include "experiment/PhysicalCondition.hpp"
#include "math/MathConstants.hpp"
#include "math/fields/Utils.hpp"
#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"
#include "mesh/Algorithms.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"
#include "mesh/Utils.hpp"
#include "meshes/plate/PlateGrid.hpp"
#include "operators/Operators.hpp"
#include "slae_generation/MatrixGeneration.hpp"
#include "visualisation/VTKFunctions.hpp"

#include "Functions.hpp"

#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SVD>
#include <unsupported/Eigen/IterativeSolvers>

#include <filesystem>
#include <iostream>

using namespace EMW;
using namespace EMW::Types;

/*
 * Сеточный аналог векторной дельта-функции, заданной на ячейках сетки
 */
Vector3c discrete_delta(const Mesh::IndexedCell &cell) {
    const Vector3c e{complex_d{1, 0}, complex_d{0, 0}, complex_d{0, 0}};
    const Mesh::point_t support{0, 0.001, 0};
    return e * Mesh::Algorithm::PointInTriangle(support, cell.getVertex()) / cell.area_;
}

Math::SurfaceVectorField getDeltaRHS(const Mesh::SurfaceMesh &surface_mesh) {
    auto rhs_field = Math::SurfaceVectorField(surface_mesh, discrete_delta);
    rhs_field.setName("delta_rhs");
    return rhs_field;
}

Math::SurfaceVectorField matrixOperatorK(const MatrixXc &A, const Math::SurfaceVectorField &field) {
    const VectorXc A_f = A * field.asSLAERHS();
    return Math::SurfaceVectorField::TangentField(field.getManifold(), A_f);
}

Math::SurfaceVectorField residual(const MatrixXc &A, const Math::SurfaceVectorField &sol,
                                  const Math::SurfaceVectorField &rhs) {
    return Math::SurfaceVectorField::TangentField(sol.getManifold(),
                                                  A * sol.asSLAERHS() - rhs.crossWithNormalField().asSLAERHS());
}

std::array<Types::Vector3d, 3> equalLocalBasis_1(const Mesh::IndexedCell &cell) {
    return {Vector3d{1, 0, 0}, Vector3d{0, 1, 0}, cell.normal};
}

Types::Vector3d collocationPoint(const Mesh::IndexedCell &cell) {
    const auto vertex = cell.getVertexAsArray();
    const Vector3d center = (1. / 3.) * (vertex[0] + vertex[1] + vertex[2]);
    return center;
}

Types::scalar multiplier(const Types::Vector3d &x) {
    if (x.norm() > 0.1)
        return 1;
    return 0;
}

Math::SurfaceVectorField solve(const MatrixXc &A, const Math::SurfaceVectorField &incidentField,
                               const scalar tolerance) {
    const VectorXc b = incidentField.asSLAERHS();

    auto method = Eigen::GMRES<MatrixXc>{};
    method.setMaxIterations(30000);
    std::cout << method.maxIterations() << std::endl;
    method.setTolerance(tolerance);
    method.set_restart(2000);

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

    method.compute(A);
    const VectorXc j_vec = VectorXc{method.solve(b)};

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::ratio<1, 1>> elapsed_seconds = end - start;
    std::cout << "Время решения GMRES: " << elapsed_seconds.count() << std::endl;

    auto j = Math::SurfaceVectorField::TangentField(incidentField.getManifold(), j_vec);
    j.setName("j");
    std::cout << "total iterations: " << method.iterations() << std::endl;
    std::cout << "tolerance: " << method.tolerance() << std::endl;
    std::cout << "total error: " << method.error() << std::endl;
    std::cout << "Info: " << static_cast<int>(method.info()) << std::endl;
    return j;
}

scalar checkBasis(const Mesh::SurfaceMesh &mesh) {
    scalar result = 0;
    for (const auto &cell : mesh.getCells()) {
        result += (cell.tau[0].cross(cell.tau[1]) - cell.normal).norm();
    }
    return result;
}

Math::SurfaceVectorField operatorK(const Math::SurfaceVectorField &field, const Types::complex_d k,
                                   const Mesh::SurfaceMesh &targetMesh) {
    const auto analytical = [field, k](const Types::Vector3d &point) -> Vector3c {
        return EMW::Operators::K1_singularityExtraction<DefiniteIntegrals::GaussLegendre::Quadrature<4, 4>>(point, k,
                                                                                                            field) +
               EMW::Operators::K0<DefiniteIntegrals::GaussLegendre::Quadrature<4>>(point, k, field);
    };

    return Math::SurfaceVectorField(targetMesh, analytical);
}

int main() {
    // треугольная сетка не пластинке
    // auto surfaceMesh = Mesh::Utils::loadTriangularMesh(2350, 4522, "1_1");
    // surfaceMesh.customLocalBasis(equalLocalBasis_1);

    int N1 = 41;
    scalar h1 = 1. / (N1 - 1);
    int N2 = 41;
    scalar h2 = 1. / (N2 - 1);
    auto surfaceMesh = Examples::Plate::generateRectangularMesh(N1, N2, h1, h2);

    surfaceMesh.setName("rect_" + std::to_string(N1) + "_x_" + std::to_string(N2));
    surfaceMesh.customLocalBasis(equalLocalBasis_1);
    const auto mesh_name = surfaceMesh.getName();

    // падающее поле на треугольной сетке
    constexpr complex_d k = 4 * Math::Constants::PI<scalar>() * complex_d{1, 0.05};

    EMW::Physics::planeWaveCase physics{Vector3d{1, 0, 0}, k, Vector3d{0, 0, -1}.normalized()};
    auto rhs_field =
        Math::SurfaceVectorField{surfaceMesh, [physics](const Mesh::point_t &point) { return physics.value(point); }};
    rhs_field.setName("constant_rhs");
    // расчет
    surfaceMesh.setName(surfaceMesh.getName() + "_" + std::to_string(k.real()) + "_+_i_" + std::to_string(k.imag()) +
                        "_" + rhs_field.getName());

    const MatrixXc MatrixK = Matrix::getMatrix(k, surfaceMesh);
    // ставится задача K[u] = m x n, которая на бесконечной плоскости
    // имеет решение вида u = (4/k^2) n x K[m].
    // в качестве m сейчас у нас дельта-функция discrete_delta_field
    // построим такое же решение, но на конечной плоскости surfaceMesh
    auto analytical_solution = (4. / (k * k)) * matrixOperatorK(MatrixK, rhs_field).normalCrossField();
    analytical_solution.setName("analytical_solution");
    std::cout << analytical_solution.getName() << std::endl;
    //    analytical_solution.multiply(multiplier);

    // рассмотрим задачу K[u] = m x n численно и решим её как обычно
    auto numerical_solution = solve(MatrixK, rhs_field.crossWithNormalField(), 1e-2);
    numerical_solution.setName("numerical_solution");
    // numerical_solution.multiply(multiplier);

    // вводим ещё одно многообразие и смотрим на нем картину полей
    auto sliceMesh = Examples::Plate::generatePlatePrimaryMesh(41, 1. / 40);
    sliceMesh.setName("sliceMesh_" + std::to_string(k.real()) + "_+_i_" + std::to_string(k.imag()) + "_" +
                      rhs_field.getName());
    auto e_from_numerical =
        operatorK(numerical_solution, k, sliceMesh) +
        Math::SurfaceVectorField{sliceMesh, [physics](const Mesh::point_t &point) { return physics.value(point); }};
    e_from_numerical.setName("e_from_numerical");
    auto e_from_analytical = operatorK(analytical_solution, k, sliceMesh) +
        Math::SurfaceVectorField{sliceMesh, [physics](const Mesh::point_t &point) { return physics.value(point); }};
    e_from_analytical.setName("e_from_analytical");
    auto e_difference = e_from_numerical - e_from_analytical;
    e_difference.setName("e_difference");

    // сохраняем результаты
    const std::string path = Pathes::studies + "plane/comparison/complex_k/weak/" + mesh_name + '/';
    std::filesystem::create_directories(path);

    VTK::united_snapshot({rhs_field, analytical_solution, numerical_solution},{}, surfaceMesh, path);
    VTK::united_snapshot({e_from_numerical, e_from_analytical, e_difference},{}, sliceMesh, path);
};
