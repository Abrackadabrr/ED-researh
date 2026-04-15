//
// Created by evgen on 12.10.2024.
//

#include "examples/pathes.hpp"
#include "experiment/PhysicalCondition.hpp"
#include "math/MathConstants.hpp"
#include "math/fields/Utils.hpp"
#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"
#include "mesh/Algorithms.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"
#include "meshes/plate/PlateGrid.hpp"
#include "operators/Operators.hpp"
#include "slae_generation/MatrixGeneration.hpp"
#include "visualisation/VTKFunctions.hpp"

#include "Functions.hpp"

#include <Eigen/Core>

using namespace EMW;
using namespace EMW::Types;

/*
 * Сеточный аналог векторной дельта-функции, заданной на ячейках сетки
 */
Vector3c discrete_delta(const Vector3d &e, const Mesh::IndexedCell &cell) {
    const Mesh::point_t support{0, 0, 0};
    return e * Mesh::Algorithm::PointInTriangle(support, cell.getVertex()) / cell.area_;
}

Math::SurfaceVectorField problemOperator(const Math::SurfaceVectorField &field, scalar k) {
    const auto analytical = [field, k](const Types::Vector3d &point) {
        return EMW::Operators::K1_singularityExtraction<DefiniteIntegrals::GaussLegendre::Quadrature<12, 12>>(point, k,
                                                                                                              field) +
               EMW::Operators::K0<DefiniteIntegrals::GaussLegendre::Quadrature<12>>(point, k, field);
    };

    return Math::SurfaceVectorField(field.getManifold(), analytical).normalCrossField();
}

std::array<Types::Vector3d, 3> equalLocalBasis_1(const Mesh::IndexedCell &cell) {
    return {Vector3d{1, 0, 0}, Vector3d{0, 1, 0}, cell.normal};
}

Types::Vector3d collocationPoint(const Mesh::IndexedCell &cell) {
    const auto vertex = cell.getVertexAsArray();
    return (1. / 3.) * (vertex[0] + vertex[1] + vertex[2]);
}

Types::scalar multiplier(const Types::Vector3d &x) {
    if (x.norm() > 0.2)
        return 1;
    return 0;
}

int main() {
    // треугольная сетка не пластинке
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "plate/triangulated/3021_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "plate/triangulated/5840_cells.csv";
    const EMW::Types::index nNodes = 3021;
    const EMW::Types::index nCells = 5840;

    int N1 = 40;
    int N2 = 40;
    scalar h1 = 1. / (N1 - 1);
    scalar h2 = 1. / (N2 - 1);

    auto surfaceMesh = Examples::Plate::generateRectangularMesh(N1, N2, h1, h2);
    surfaceMesh.setName("rectangularMesh_" + std::to_string(N1) + "_x_" + std::to_string(N2));
    surfaceMesh.customCollocationPpoints(collocationPoint);

    // auto surfaceMesh = Mesh::SurfaceMesh{EMW::Parser::parseMesh(nodesFile, cellsFile, nNodes, nCells)};
    // surfaceMesh.setName("triangular_mesh_" + std::to_string(nCells));

    // подправляем сетку для одинаковых базисов
    surfaceMesh.customLocalBasis(equalLocalBasis_1);

    const Vector3d e{0, 1, 0};
    // поле на треугольной сетке
    EMW::Physics::planeWaveCase physics{Vector3d{0, 1, 0}.normalized(), 4 * Math::Constants::PI<scalar>(),
                                        Vector3d{0, 0, 1}.normalized()};
    const auto initial_field_function = [physics](const Mesh::point_t &point) -> Vector3c {
        return physics.value(point);
    };

    // след падающего поля на расчетной поверхности
    Math::SurfaceVectorField incidentField(surfaceMesh, initial_field_function);
    incidentField.setName("incidentField");

    // дискретизованный оператор K, примененный полю на сетке
    auto first = problemOperator(incidentField, physics.k);
    first.setName("operator_computation");

    auto second = Math::SurfaceVectorField::TangentField(surfaceMesh,
        Matrix::getMatrix(physics.k, surfaceMesh) * incidentField.asSLAERHS());
    second.setName("matrix_computation");

    auto diff = first - second;
    diff.setName("diff");

    // сохраняем результаты
    VTK::united_snapshot({incidentField, first, second, diff},
        {Math::FieldUtils::relativeError(second, first)}, surfaceMesh,
        Pathes::studies + "plane/analytical_solution/operator_conditions/");
}
