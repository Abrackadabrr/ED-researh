//
// Created by evgen on 11.06.2025.
//

#include "experiment/PhysicalCondition.hpp"
#include "math/MathConstants.hpp"
#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"
#include "mesh/MeshTypes.hpp"
#include "mesh/Parser.hpp"
#include "mesh/SurfaceMesh.hpp"
#include "mesh/Utils.hpp"
#include "slae_generation/MatrixGeneration.hpp"
#include "types/Types.hpp"
#include "visualisation/VTKFunctions.hpp"

#include "operators/OperatorK.hpp"
#include "slae_generation/MatrixGeneration.hpp"

#include "../Solve.hpp"

#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <fstream>
#include <iostream>
#include <ranges>
#include <unsupported/Eigen/IterativeSolvers>

using namespace EMW;
using namespace EMW::Types;

inline Types::Vector3c operatorK_in_point(const Math::SurfaceVectorField &field, const Types::complex_d k,
                                          const Mesh::point_t &point) {
    return EMW::OperatorK::K1<DefiniteIntegrals::GaussLegendre::Quadrature<4, 4>>(point, k, field) +
           EMW::OperatorK::K0<DefiniteIntegrals::GaussLegendre::Quadrature<4>>(point, k, field);
}

inline Types::Vector3c getE_in_point(const Math::SurfaceVectorField &j_e, const Types::complex_d k,
                                     const Mesh::point_t &point) {
    const Types::complex_d mul = Math::Constants::i / k;
    const auto value = operatorK_in_point(j_e, k, point);
    // std::cout << value << std::endl;
    return value;
}

int main() {
    const std::string nodesFile =
        "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/sphere/tri/412_nodes.csv";
    const std::string cellsFile =
        "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/sphere/tri/820_cells.csv";

    const auto parser_out = EMW::Parser::parse_mesh_without_tag(nodesFile, cellsFile);
    auto mesh_base =
        Mesh::SurfaceMesh(parser_out.nodes, parser_out.cells, [&](const Mesh::IndexedCell::nodes_t& nodes, const Containers::vector<Mesh::point_t>& full_points) -> Vector3d {
                              return (1. / 3.) * (full_points[nodes[0]] + full_points[nodes[1]] + full_points[nodes[2]]);
                          });
    mesh_base.setName("sphere");

    // физика
    const Types::scalar freq = 0.1;
    const Types::complex_d k{(2 * Math::Constants::PI<Types::scalar>() * 1e9 * freq) / Math::Constants::c, 0};
    std::cout << k.real() << std::endl;

    EMW::Physics::planeWaveCase physics(Vector3d{0, 1, 0}.normalized(),  // polarization
                                        k,                               // wave figure
                                        Vector3d{-1, 0, 0}.normalized()); // wave unit vector
    // смотрим след поля на поверхности
    const auto E_0_field =
        Math::SurfaceVectorField{mesh_base, [&physics](const Mesh::point_t &p) { return physics.value(p); }};
    const Types::VectorXc b = -1 * E_0_field.asVector();

    // расчет
    const MatrixXc A = EMW::Matrix::getMatrixK(k, mesh_base);
    const auto j_vec = Research::solve<Eigen::GMRES>(A, b, 20000, 1e-5);

    // закидываем ток как поле на поверхности
    const Math::SurfaceVectorField j_e = Math::SurfaceVectorField::TangentField(mesh_base, j_vec, "j_e");

    // расчитываем поле вокруг заданной геометрии
    const std::string path = "/home/evgen/Work/INM_RAS/Fidesys/example_calculations/";
#define FIELD_CALCULATION 0
#if FIELD_CALCULATION
    // Рисуем картину поля в плоскости y = 0
    int k1 = 199;
    Types::scalar h1 = 4. / (k1 - 1);
    int k2 = 199;
    Types::scalar h2 = 4. / (k2 - 1);

    std::vector<Mesh::point_t> points;
    points.reserve(k1 * k2);
    Mesh::Utils::cartesian_product_unevenYZ(std::ranges::views::iota(0, k2), std::ranges::views::iota(0, k1),
                                            std::back_inserter(points), k2, k1, h2, h1);

    std::cout << "Surrounding Mesh Constructed" << std::endl;

    Containers::vector<Types::Vector3c> calculated_field;
    calculated_field.resize(points.size());

#pragma omp parallel for num_threads(14) schedule(dynamic)
    for (int i = 0; i < points.size(); ++i) {
        calculated_field[i] = getE_in_point(j_e, k, points[i]) + physics.value(points[i]);
    }

    // отрисовка полученного поля
    std::transform(calculated_field.begin(), calculated_field.end(), calculated_field.begin(), [](auto &&v) {
        if (v.norm() > 10)
            return Types::Vector3c{{0, 0}, {0, 0}, {0, 0}};
        return v;
    });
    VTK::field_in_points_snapshot({calculated_field}, {}, {"E"}, {}, points, "surrounding_mesh", path);
#endif
    // рисуем поле токов
    VTK::united_snapshot<Math::SurfaceScalarField<Types::complex_d>>({}, {j_e}, mesh_base, path);
}
