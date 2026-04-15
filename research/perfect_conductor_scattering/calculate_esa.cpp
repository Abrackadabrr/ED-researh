//
// Created by evgen on 11.06.2025.
//

#include "mesh/Parser.hpp"
#include "types/Types.hpp"
#include "mesh/SurfaceMesh.hpp"
#include "mesh/MeshTypes.hpp"
#include "visualisation/VTKFunctions.hpp"
#include "mesh/Parser.hpp"
#include "slae_generation/MatrixGeneration.hpp"
#include "math/MathConstants.hpp"
#include "examples/pathes.hpp"
#include "experiment/PhysicalCondition.hpp"
#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"
#include "mesh/Utils.hpp"
#include "Utils.hpp"

#include "operators/OperatorK.hpp"
#include "slae_generation/MatrixGeneration.hpp"
#include "experiment/ESA.hpp"

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
    const std::string nodesFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/plate/triangulated/1_1/nodes/1440_nodes.csv";
    const std::string cellsFile = "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/meshes/plate/triangulated/1_1/cells/2742_cells.csv";

    const auto parser_out = EMW::Parser::parse_mesh_without_tag(nodesFile, cellsFile);
    auto mesh_base = Mesh::SurfaceMesh{parser_out.nodes, parser_out.cells};
    mesh_base.setName("Plane");

    // физика
    const Types::scalar lambda = 0.5;
    const Types::complex_d k{2 * Math::Constants::PI<scalar>() / lambda, 0};
    std::cout << k.real() << std::endl;
    // расчет матрицы системы
    const MatrixXc A = EMW::Matrix::getMatrixK(k, mesh_base);

    // цикл по векторам правой части
    int samples = 180;
    Containers::vector_d esas;
    esas.resize(samples);
    Containers::vector_d angles;
    angles.resize(samples);

    int tasks_done = 0;

#pragma omp parallel for schedule(dynamic) num_threads(14) firstprivate(mesh_base)
    for (int i = 0; i < samples; i++) {
        // угол в радианах
        Types::scalar angle = i * 2 * Math::Constants::PI<scalar>() / (samples);
        // вектор k
        Types::Vector3d k_vec = {-std::cos(angle), -std::sin(angle), 0};
        // вектор E
        // задаем падающую волну в виде плоской волны
        const Vector3d E0_V{0, 0, 1};
        const Vector3d E0_H = Vector3d{std::sin(angle), -std::cos(angle), 0}.normalized();
        EMW::Physics::planeWaveCase physics(E0_V,     // polarization
                                            k,  // wave figure
                                            k_vec);   // wave unit vector
        // смотрим след поля на поверхности
        const auto E_0_field = Math::SurfaceVectorField{mesh_base, [&physics](const Mesh::point_t& p) {return physics.value(p);}};
        const auto b = -E_0_field.asVector();
        // решаем СЛАУ
        const auto j_vec = Research::solve<Eigen::GMRES>(A, b, 2000, 1e-2);
        // закидываем ток как поле на поверхности
        const Math::SurfaceVectorField j_e = Math::SurfaceVectorField::TangentField(mesh_base, j_vec, "j_e");

        int rank = omp_get_thread_num();
        std::cout << "#" + std::to_string(rank)<< std::endl;
        esas[i] = ESA::calculateESA(-k_vec, k, j_e);
        angles[i] = angle;
        tasks_done++;
        std::cout << "Tasks done: " << tasks_done << std::endl << std::endl;
    }

    std::ofstream sigmaFile("/home/evgen/Education/MasterDegree/thesis/results/mobius/sigma_back.csv");

    EMW::Utils::to_csv(esas, angles, "sigma", "angle", sigmaFile);
}
