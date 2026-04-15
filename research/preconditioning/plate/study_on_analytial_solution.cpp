//
// Created by evgen on 29.07.24.
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
#include <Eigen/IterativeLinearSolvers>
#include <iostream>
#include <unsupported/Eigen/IterativeSolvers>

using namespace EMW;
using namespace EMW::Types;

/*
 * Сеточный аналог векторной дельта-функции, заданной на ячейках сетки
 */
Vector3c discrete_delta(const Vector3d &e, const Mesh::IndexedCell &cell) {
    const Mesh::point_t support{0, 0, 0};
    return e * Mesh::Algorithm::PointInTriangle(support, cell.getVertex()) / cell.area_;
}

Math::SurfaceVectorField operatorK(const Math::SurfaceVectorField &field, const Types::complex_d k) {
    const auto analytical = [field, k](const Types::Vector3d &point) -> Vector3c {
        return EMW::Operators::K1_singularityExtraction<DefiniteIntegrals::GaussLegendre::Quadrature<8, 8>>(point, k,
                                                                                                            field) +
               EMW::Operators::K0<DefiniteIntegrals::GaussLegendre::Quadrature<8>>(point, k, field);
    };

    return Math::SurfaceVectorField(field.getManifold(), analytical);
}

std::array<Types::Vector3d, 3> equalLocalBasis_1(const Mesh::IndexedCell &cell) {
    return {Vector3d{1, 0, 0}, Vector3d{0, 1, 0}, cell.normal};
}

Types::Vector3d collocationPoint(const Mesh::IndexedCell &cell) {
    const auto vertex = cell.getVertexAsArray();
    return (1. / 3.) * (vertex[0] + vertex[1] + vertex[2]);
}

Types::scalar multiplier(const Types::Vector3d &x) {
    if (x.norm() > 0.02)
        return 1;
    return 0;
}

/*
 * Эта программа сравнивает аналитическую формулу, которая получается, если в правой части стоит дельта-функция,
 * с непосредственным расчетом оператора K от дельта-функции.
 * 1) Для сетки из кватратов:
 * Решения получаются схожими в мнимой части значений,
 * при этом в некоторой окрестности носителя дельты имеется относительное расхождение порядка 0.2
 * Связано ли это в дискретизацией оператора пока что неясно
 *
 * 2) Для сетки из треугольников:
 * -- Рассматриваю сетку 2350_4522
 * За исключением областей с малыми значениями нормы вектора (где-то меньше 0.9) функция, полученная с помощью
 * применения дискретизованного оператора, совпадает с аналитическим выражением с точностью 0.08. Расхождение на области
 * малости номр вектора можно попытаться объяснить тем, что функция в этой точке есть сумма большого числа слагаемых
 * близких по модулю, но противоположных по знаку, в связи с чем получается неустойчивость при сложении в арифметике
 * компьютера. Это можно решить тем, чтобы сначала складывать "положительные значения", а затем отрицательные. Тогда
 * вычисления станут более устойчивыми, и, возможно, решения будут более точно сходиться.
 *
 * -- Рассмотрим сетку более мелкую: 7045_13780
 * Ситуация аналогичная, кроме точек с нормой меньше 0.3 относительная ошибка с аналитикой составляет где-то 0.08
 * Думается, что это все из-за вычитания близких по норме чиселок.
 */
int main() {
    // треугольная сетка не пластинке
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "plate/triangulated/1_1/nodes/7045_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/"
                                  "plate/triangulated/1_1/cells/13780_cells.csv";
    const EMW::Types::index nNodes = 7045;
    const EMW::Types::index nCells = 13780;

    auto surfaceMesh = Mesh::SurfaceMesh{EMW::Parser::parseMesh(nodesFile, cellsFile, nNodes, nCells)};
    surfaceMesh.setName("triangular_mesh_" + std::to_string(nCells));
    surfaceMesh.customLocalBasis(equalLocalBasis_1);
    surfaceMesh.customCollocationPpoints(collocationPoint);

    // падающее поле на треугольной сетке
    const Vector3d e{0, 1, 0};
    auto discrete_delta_field =
        Math::SurfaceVectorField(surfaceMesh, [e](const Mesh::IndexedCell &cell) { return discrete_delta(e, cell); });
    discrete_delta_field.setName("delta_discrete_delta");

    // расчет
    const Types::complex_d k = 4 * Math::Constants::PI<scalar>() * complex_d{1, 0.0} + complex_d{0, 0.05};
    std::cout << k*k << std::endl;

    // дискретизованный оператор K, примененный полю на сетке
    auto analytical_solution_numerical = (4. / (k*k)) * operatorK(discrete_delta_field, k);
    analytical_solution_numerical.setName("K_numerical");
    // std::cout << analytical_solution_numerical.getName() << std::endl;
    // analytical_solution_numerical.multiply(multiplier);

    // аналитическое применение оператора К к дельта функции на сетке
    auto analytical_solution_analytical =
        (4. / (k*k)) * Math::SurfaceVectorField(surfaceMesh,
                                 [e, k](const Mesh::IndexedCell &cell) { return kernelFromPaper(e, cell, k); });
    analytical_solution_analytical.setName("K_analytical");
    // analytical_solution_analytical.multiply(multiplier);

    // расчитываем разницу между полями
    auto analytical_difference = analytical_solution_numerical - analytical_solution_analytical;
    analytical_difference.setName("difference");
    // и относительную ошибку по каждой из компонент
    auto relativeError = Math::FieldUtils::relativeError(analytical_solution_numerical, analytical_solution_analytical);
    relativeError.setName("relativeError");

    // сохраняем результаты
    VTK::united_snapshot({discrete_delta_field, analytical_solution_numerical,
        analytical_solution_analytical, analytical_difference},
        {relativeError}, surfaceMesh,
        Pathes::studies + "plane/analytical_solution/triangular/");
}
