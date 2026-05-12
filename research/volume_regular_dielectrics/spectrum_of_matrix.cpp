//
// Created by evgen on 15.04.2026.
//
#include "types/Types.hpp"

#include "mesh/volume_mesh/CubeMeshWithData.hpp"

#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "experiment/ESA.hpp"
#include "experiment/PhysicalCondition.hpp"

#include "visualisation/include/VTKFunctions.hpp"

#include "math/fields/Utils.hpp"
#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"

#include "Utils.hpp"

#include <Eigen/Eigenvalues>

#include <iostream>

using namespace EMW;

constexpr Types::scalar SPHERE_RADUIS = 0.5;
constexpr Types::scalar CUBE_LENGTH = 0.5;

Types::scalar homo_sphere(const Types::point_t &x) { return x.norm() < SPHERE_RADUIS ? 2.56 : 1; }

Types::scalar permittivity_distribution_cube(const Types::point_t &x) {
    return 3 * ((std::abs(x.x()) < CUBE_LENGTH / 2) && (std::abs(x.y()) < CUBE_LENGTH / 2) &&
                (std::abs(x.z()) < CUBE_LENGTH / 2)) +
           1;
}

template <typename Container1>
void to_csv(const Container1& cont1, const std::string& name1,
            std::ostream& str, char delimiter = ',')
{
    str << name1 << delimiter << "\n";
    for (int i = 0; i < cont1.size(); i++)
    {
        str << cont1[i] << delimiter << '\n';
    }
}


int main() {
    Eigen::setNbThreads(1);
    openblas_set_num_threads(1);
    // 1. Рисуем сетку
    constexpr Types::scalar cube_length = 1;
    constexpr Types::index Nx = 11;
    constexpr Types::index Ny = 11;
    constexpr Types::index Nz = 11;
    constexpr Types::scalar mesh_one_axis_size = cube_length / (Nx - 1);
    Mesh::VolumeMesh::CubeMeshWithData mesh{Types::point_t{-cube_length / 2, -cube_length / 2, -cube_length / 2},
                                            (Nx - 1) * mesh_one_axis_size,
                                            (Ny - 1) * mesh_one_axis_size,
                                            (Nz - 1) * mesh_one_axis_size,
                                            Nx,
                                            Ny,
                                            Nz};
    mesh.setName("sphere_21");
    const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();
    const Types::scalar basis_fn_module = 1. / sqrt(cube_measure);
    // Настраиваем диэлектрическую проницаемость
    mesh.smoothScalarData<DefiniteIntegrals::GaussLegendre::Quadrature<4, 4, 4>>("eps", homo_sphere);

    // 2. Параметры падающей волны
    constexpr double freq = 1; // GHz
    constexpr Types::complex_d k{Physics::get_k_on_frquency(freq), 0.};
    Physics::planeWaveCase incident_field{Types::Vector3d{0, 1, 0}, k, Types::Vector3d{1, 0, 0}};
    std::cout << "Длина волны в свободном пространстве = " << 2 * M_PI / k.real() << std::endl;

    // 3. Галеркинская проекция правой части
    Operators::Volume::ProjectorOnMesh proj{mesh};
    auto rhs = proj([incident_field](Types::point_t p) { return incident_field.value(p); });
    // Поправляем правую часть
    Types::VectorXc b = rhs * basis_fn_module;

    // 4. Галеркинская проекция оператора (две матрицы: точная и апроксимированная)
    size_t nx, ny, nz;
    nx = 1;
    ny = 1;
    nz = 1;
    Operators::Volume::operator_K_over_cube_mesh operator_K{k, mesh};
    auto [mat_compressed, perm] = operator_K.compute_galerkin_matrix_custom_blocksize(nx, ny, nz, basis_fn_module);
    auto dense_matrix = mat_compressed.to_dense();
    // Сразу модифицируем правую часть (матрица перестановки)
    b = perm * b;
    // Собираем матрицу (eps - 1) как вектор из значений
    Types::VectorXc diag_eps = Types::VectorXc::Zero(3 * mesh.getCells().size());
    Types::VectorXc diag_eps_m_1 = Types::VectorXc::Zero(3 * mesh.getCells().size());
    const auto eps_data = mesh.getScalarData("eps");
    for (size_t idx = 0; idx < mesh.getCells().size(); ++idx) {
        diag_eps_m_1[3 * idx] = eps_data[idx] - 1.;
        diag_eps_m_1[3 * idx + 1] = eps_data[idx] - 1.;
        diag_eps_m_1[3 * idx + 2] = eps_data[idx] - 1.;
        diag_eps[3 * idx] = eps_data[idx];
        diag_eps[3 * idx + 1] = eps_data[idx];
        diag_eps[3 * idx + 2] = eps_data[idx];
    }
    // И его тоже переставляем
    diag_eps = perm * diag_eps;
    diag_eps_m_1 = perm * diag_eps_m_1;

    // 5. Проводим анализ матрицы
    std::cout << Utils::get_memory_usage(mat_compressed) << std::endl;
    std::cout << "Matrix size: " << mat_compressed.rows() << std::endl;

    // Находим собственные числа матрицы системы
    Eigen::setNbThreads(10);
    const Types::MatrixXc full_matrix = Types::DiagonalMatrixXc{diag_eps.cwiseInverse().cwiseSqrt()} * (Types::MatrixXc::Identity(mat_compressed.rows(), mat_compressed.cols()) -
        dense_matrix * Types::DiagonalMatrixXc{diag_eps_m_1}) * Types::DiagonalMatrixXc{diag_eps.cwiseInverse().cwiseSqrt()};
    Eigen::ComplexSchur<Types::MatrixXc> schur_dec{};
    schur_dec.compute(full_matrix);
    const auto eigen_values = schur_dec.matrixT().diagonal();
    // Пишем результат
    const std::string path = "/home/evgen/Education/MasterDegree/thesis/ED-researh/research/volume_regular_dielectrics/";
    std::ofstream ev_file{path + "sqrt_jvie_ev_" + std::to_string(Nx) + ".csv"};
    ev_file.precision(18);
    const Types::VectorXd real = eigen_values.real();
    const Types::VectorXd imag = eigen_values.imag();
    Utils::to_csv(real, imag, "real", "imag", ev_file);
};

// ВАЖНО ДЛЯ ЭФФЕКТИАВНОСТИ РАБОТЫ
// БЛОЧНОГО УМНОЖЕНИЯ тёплицевой МАТРИЦЫ НА ВЕКТОР: нужно посмотреть как ведёт себя ранг, если жать не поблочно
// а целиком прямо строку и столбец в тёплицевой матрице первого уровня.Тогда ябуду жать не мелки блоки 192 * 192
// а блоки вида (Nx * 192) x 192, и это поможет эффективнее умножать, чем на каждый блок с рангом 1.

// При этом диагональные блоки нужно жать через строки и столбца + блок на диагонали, а внедиагональные
// просто через строку и столбец по отдельности. Наверное отсюда можно будет много выиграть, так как
// не нужно будет перескакивать с блока на блок при умножении, а сразу делать "эффективно".

// Нужно реализовать и забенчмаркать.
