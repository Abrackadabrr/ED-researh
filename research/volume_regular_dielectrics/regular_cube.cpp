//
// Created by evgen on 10.02.2026.
//

#include "types/Types.hpp"

#include "mesh/volume_mesh/CubeMeshWithData.hpp"

#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "experiment/PhysicalCondition.hpp"

#include "../Solve.hpp"

#include "VTKFunctions.hpp"

#include "math/fields/Utils.hpp"
#include "math/matrix/iterative_solvers_coverage/MatrixTraits.hpp"

#include "MatrixReplacement.hpp"
#include "MatrixTraits.hpp"

#include <iostream>

using namespace EMW;

int main() {
    // Параметры куба
    constexpr Types::scalar cube_length = 0.0125;
    constexpr Types::index Nx = 5;
    constexpr Types::index Ny = static_cast<size_t>(1 / cube_length) + 1;
    constexpr Types::index Nz = Ny;

    // Сетка
    Mesh::VolumeMesh::CubeMeshWithData mesh{Types::point_t{-cube_length, -0.5, -0.5}, (Nx - 1) * cube_length,
                                            (Ny - 1) * cube_length, (Nz - 1) * (cube_length), Nx, Ny, Nz};

    const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();
    // Квадрат L2 нормы базисных функций
    std::cout << "Квадрат L2 нормы базисных функций = " << cube_measure << std::endl;

    // Параметры падающей волны
    constexpr double freq = 0.3; // GGz
    constexpr Types::complex_d k{2 * M_PI, 0.};
    Physics::planeWaveCase incident_field{Types::Vector3d{0, 1, 0}, k, Types::Vector3d{1, 0, 0}};

    std::cout << "Длина волны в свободном пространстве = " << 2 * M_PI / k.real() << std::endl;

    // Галеркинская проекция правой части на сетку (правая часть линейной системы)
    Operators::Volume::ProjectorOnMesh proj{mesh};
    auto rhs = proj([incident_field](Types::point_t p) { return incident_field.value(p); });
    std::cout << "RHS size: " << rhs.rows() << std::endl;

    // Поправляем правую часть
    const Types::VectorXc b = rhs / std::sqrt(cube_measure);

    // Матрицу собираем и решаем линейную систему
    Operators::Volume::operator_K_over_cube_mesh operator_K{k, mesh};
#define DENSE 0
#if DENSE
    std::cout << "Calculation of Full Matrix" << std::endl;
    auto mat = operator_K.get_galerkin_matrix();
    // Собираем матрицу системы
    mat = mat / cube_measure - Types::MatrixXc::Identity(mat.rows(), mat.cols());
    std::cout << "Matrix (" << mat.rows() << ", " << mat.cols() << ')' << std::endl;
    // Решаем линейную систему
    const auto solution = Research::solve<Eigen::GMRES>(mat, b, 1000,
                                                        1e-8);
    // Пишем правильное имя для сетки
    mesh.setName("full_calc");
#else
    std::cout << "Calculation of Toelpitz Matrix" << std::endl;

    // Галеркинская проекция на сетку
    auto mat = operator_K.compute_galerkin_matrix(std::sqrt(cube_measure));
    // Поле диэлектрической проницаемости
    Types::VectorXc diag_eps = Types::VectorXc::Zero(3 * mesh.getCells().size());
    for (size_t idx = 0; idx < mesh.getCells().size(); ++idx) {
        diag_eps[3 * idx] = 1.;
        diag_eps[3 * idx + 1] = 1.;
        diag_eps[3 * idx + 2] = 1.;
    }
    const auto eps_matrix = Eigen::DiagonalMatrix<Types::complex_d, Eigen::Dynamic>{diag_eps};
    // 4. Решаем систему
    const Math::LinAgl::Matrix::Wrappers::VolumeOperatorMatrixReplacement A{mat, eps_matrix};
    const auto solution = Research::solve<Eigen::GMRES>(A, b, 1000, 1e-8);

    // Пишем правильное имя для сетки
    mesh.setName("toep_calc");
#endif

    // Преобразовываем в векторное поле на ячейках
    std::vector<Types::Vector3c> field_on_mesh;
    field_on_mesh.reserve(mesh.getCells().size());
    for (size_t idx = 0; idx < mesh.getCells().size(); ++idx) {
        field_on_mesh.emplace_back(solution[3 * idx + 0] / std::sqrt(cube_measure),
                                   solution[3 * idx + 1] / std::sqrt(cube_measure),
                                   solution[3 * idx + 2] / std::sqrt(cube_measure));
    }

    // Добавляем поле
    mesh.setVectorData("solution", std::move(field_on_mesh));

    // Отрисовываем результат
    VTK::volume_mesh_withdata_snapshot(
        mesh,
        "/home/evgen/Education/MasterDegree/thesis/Electromagnetic-Waves-Scattering/research/volume_regular_dielectrics/");

#define CALCULATE_FIELD 0

#if CALCULATE_FIELD
    // Делаем множество точек вокруг куба в плоскости Оху
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
}
