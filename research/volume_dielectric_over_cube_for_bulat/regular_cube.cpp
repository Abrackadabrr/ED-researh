//
// Created by evgen on 10.02.2026.
//

#include "types/Types.hpp"

#include "mesh/volume_mesh/CubeMeshWithData.hpp"

#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "experiment/PhysicalCondition.hpp"

#include "VTKFunctions.hpp"

#include "math/fields/Utils.hpp"

#include <iostream>

using namespace EMW;

// Диэлектрическая проницаемость как функция точки
constexpr double SPHERE_RADUIS = 1;

Types::scalar permittivity_distribution(const Types::point_t &x) {
    return x.norm() < SPHERE_RADUIS ? sqrt(2 - x.squaredNorm()) : 1;
}


int main() {
    // Параметры параллелограмма (длина сторон и количество ячеек разбиения на сторону)
    constexpr Types::scalar cube_length_x = 1;
    constexpr Types::scalar cube_length_y = 1;
    constexpr Types::scalar cube_length_z = 1;
    constexpr Types::index Nx = 10;
    constexpr Types::index Ny = 10;
    constexpr Types::index Nz = 10;

    // Строит сетку на кубе. Нумерация кубов в сетке (i, j, k) = (x, y, z)
    // сначала по i, потом по j и в конце по k
    Mesh::VolumeMesh::CubeMeshWithData mesh{
        Types::point_t{-cube_length_x / 2, -cube_length_y / 2, -cube_length_z / 2},
        // угол куба с минимальными координатами
        cube_length_x, cube_length_y, cube_length_z, // длины сторон куба
        Nx, Ny, Nz                                   // количество точек разбиения на стороне куба
    };
    mesh.setName("name_is_used_only_for_vtk_data_snapshot");

    // Норма базисных функц ий (нужен для построения матрицы)
    const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();
    const Types::scalar basis_fn_module = 1. / sqrt(cube_measure);

    // Параметры падающей волны
    constexpr double freq = 0.3; // частота в GGz
    constexpr Types::complex_d k{Physics::get_k_on_frquency(freq), 0.};
    std::cout << "Длина волны в свободном пространстве = " << 2 * M_PI / k.real() << std::endl;

    // Собираем матрицу методом Галёркина
    Operators::Volume::operator_K_over_cube_mesh operator_K{k, mesh};
    auto mat = operator_K.compute_galerkin_matrix(basis_fn_module);

    // Собираем правую часть (в зависимости от нумерации элементов на сетке)
    Physics::planeWaveCase incident_field{Types::Vector3d{0, 1, 0}, k, Types::Vector3d{1, 0, 0}};
    Operators::Volume::ProjectorOnMesh proj{mesh};
    auto rhs = proj([incident_field](Types::point_t p) { return incident_field.value(p); });
    const Types::VectorXc b = rhs / std::sqrt(cube_measure);

    // Задаём поле диэлектрической проницаемости как среднее по объёму
    // Шаблонный параметр -- это порядок квадратуры
    mesh.smoothScalarData<DefiniteIntegrals::GaussLegendre::Quadrature<4, 4, 4>>("eps", permittivity_distribution);
    // Формируем диагональ в матрице E в соответствии с нумерацией.
    auto eps_vec = mesh.getScalarData("eps");

    // Пишем в файл
    // 1. Диагональ матрицы E
    std::ofstream eps_file("epsilon_diag.txt");
    eps_file.precision(17);
    std::ostream_iterator<std::complex<double>> eps_iter_file{eps_file, ",\n"};
    std::copy(eps_vec.begin(), eps_vec.end(), eps_iter_file);

    // 2. Правая часть системы уравнений
    std::ofstream rhs_file("rhs_vector.txt");
    rhs_file.precision(17);
    std::ostream_iterator<std::complex<double>> rhs_iter_file{rhs_file, ",\n"};
    std::copy(b.begin(), b.end(), rhs_iter_file);

    // 3. Пишем матрицу
    std::ofstream m_file("matrix.txt");
    std::ofstream m_file_real("matrix_real_part.txt");
    std::ofstream m_file_imag("matrix_imag_part.txt");
    m_file.precision(17);
    m_file_real.precision(17);
    m_file_imag.precision(17);
    auto dense_matrix = mat.to_dense();
    for (auto i = 0; i < dense_matrix.rows(); ++i) {
        for (auto &&elem : dense_matrix.row(i)) {
            m_file << elem << ',';
            m_file_real << elem.real() << ',';
            m_file_imag << elem.imag() << ',';
        }
        m_file << "\n";
        m_file_real << "\n";
        m_file_imag << "\n";
    }

    // Если вдруг хочется посмотреть на объект, то
    // отрисовываем результат в формат vtu, можно поглазеть в paraview
    VTK::volume_mesh_withdata_snapshot(
        mesh,
        "path_to_directory");
}
