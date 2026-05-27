//
// Created by evgen on 22.03.2026.
//

#include "EMW/types/Types.hpp"
#include "MatrixReplacement.hpp"

#include "mesh/volume_mesh/CubeMeshWithData.hpp"

#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "experiment/PhysicalCondition.hpp"

#include "math/fields/Utils.hpp"
#include "math/fourier/TripleToeplitz3x3Fourier.hpp"

#include "../Solve.hpp"

#include "Utils.hpp"

#include "MatrixReplacement.hpp"
#include "MatrixTraits.hpp"

#include <fstream>
#include <iostream>

using namespace EMW;

template <typename ForwardIterator>
void save_complex_vector_binary(const ForwardIterator &begin, const ForwardIterator &end, const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);

    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    const auto size = static_cast<std::uint64_t>(std::distance(begin, end));
    out.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (auto it = begin; it != end; ++it) {
        auto &&z = *it;
        const double re = z.real();
        const double im = z.imag();

        out.write(reinterpret_cast<const char *>(&re), sizeof(re));
        out.write(reinterpret_cast<const char *>(&im), sizeof(im));
    }

    if (!out) {
        throw std::runtime_error("Error while writing file: " + filename);
    }
}

int main() {
    // Настроечные параметры расчета
    constexpr Types::index Nx = 16; // количество кубов в сетке на каждой стороне
    constexpr Types::index nx = 4;  // на группы по сколько кубов подразбивать
    constexpr Types::complex_d k{1, 0.};        // волновое число падающей волны
    constexpr Types::complex_d epsilon{2., 0.}; // диэлектрическая проницаемость куба

    // 1. Рисуем сетку
    constexpr Types::scalar cube_length = 1;

    constexpr Types::index Ny = Nx;
    constexpr Types::index Nz = Nx;
    constexpr Types::index ny = nx;
    constexpr Types::index nz = nx;
    constexpr Types::scalar mesh_one_axis_size = cube_length / Nx;
    Mesh::VolumeMesh::CubeMeshWithData mesh{Types::point_t{-cube_length / 2, -cube_length / 2, -cube_length / 2},
                                            Nx * mesh_one_axis_size,
                                            Ny * mesh_one_axis_size,
                                            Nz * mesh_one_axis_size,
                                            Nx + 1,
                                            Ny + 1,
                                            Nz + 1};
    const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();
    const Types::scalar basis_fn_module = 1. / sqrt(cube_measure);

    // 2. Параметры падающей волны
    Physics::planeWaveCase incident_field{Types::Vector3d{0, 1, 0}, k, Types::Vector3d{1, 0, 0}};
    std::cout << "Длина волны в свободном пространстве = " << 2 * M_PI / k.real() << std::endl;
    std::cout << "Для хорошей аппроксимации, это должно быть >= 7: lambda_inside_cube / mesh.h = "
              << 2 * M_PI / (epsilon.real() * k.real() * mesh_one_axis_size) << std::endl;

    // 3. Галеркинская проекция правой части
    Operators::Volume::ProjectorOnMesh proj{mesh};
    auto rhs = proj([incident_field](Types::point_t p) { return incident_field.value(p); });
    // Поправляем правую часть
    Types::VectorXc b = rhs * (-basis_fn_module);

    std::cout << "rhs assembled" << std::endl;

    // 4. Галеркинская проекция оператора

    // Собираем интегральную часть
    Operators::Volume::operator_K_over_cube_mesh operator_K{k, mesh};
    // настройка точности адаптивного интегрирования (выкрутил ручки, чтобы считалось точно)
    operator_K.set_tolerances(1e-8, 1e-23);
    operator_K.set_adaptive_integration_max_levels({40, 20, 20, 10});
    auto [matrix, permutation] = operator_K.compute_galerkin_matrix_custom_blocksize(nx, ny, nz, basis_fn_module);
    // Добавляем диагональную матрицу (итоговая матрица имеет вид K - \gamma^{-1} * I)
    matrix.get_block(0, 0).get_block(0, 0).get_block(0, 0).diagonal() -= Types::VectorXc::Ones(3 * nx) / (epsilon - 1.);

    std::cout << "matrix assembled" << std::endl;

    // 5. Решаем линейную систему (тут используется быстрая решалка)
    // Меняем правую часть согласно матрицы перестановки (исходно правая часть собирается под непереставленную матрицу)
    b = permutation * b;
    // Собираем матрицу без подразбиений
    auto pure_triple_toeplitz = operator_K.compute_galerkin_matrix(basis_fn_module);
    pure_triple_toeplitz.get_block(0, 0).get_block(0, 0).get_block(0, 0).diagonal() -=
        Types::VectorXc::Ones(3) / (epsilon - 1.);
    // Используем фурье-умножалку
    using fourier_t = Math::Fourier::TripleToeplitz3x3FourierParallel<Types::complex_d>;
    auto fourier = fourier_t(pure_triple_toeplitz);

    // Настройки GMRES
    const size_t max_inters = 1000;
    const size_t tolerance = 1e-6;
    const size_t restart_n = 100;
    auto solution =
        Research::solve<Eigen::GMRES>(Math::LinAgl::Matrix::Wrappers::VolumeMatrixReplacement{fourier, permutation}, b,
                                      max_inters, tolerance, restart_n);

    // 6. Дампим все артефакты в файл
    // Сборка матрицы в виде вектора с правильной индексацией
    std::vector<Types::complex_d> vectorized_matrix{};
    vectorized_matrix.reserve(matrix.rows() * matrix.cols());
    for (size_t i3 = 0; i3 < Nz / nz; ++i3)
        for (size_t j3 = 0; j3 < Nz / nz; ++j3)
            for (size_t i2 = 0; i2 < Ny / ny; ++i2)
                for (size_t j2 = 0; j2 < Ny / ny; ++j2)
                    for (size_t i1 = 0; i1 < Nx / nx; ++i1)
                        for (size_t j1 = 0; j1 < Nx / nx; ++j1) {
                            auto cur_block = matrix.get_block(i3, j3).get_block(i2, j2).get_block(i1, j1);
                            for (size_t k3 = 0; k3 < cur_block.rows(); ++k3)
                                for (size_t m3 = 0; m3 < cur_block.cols(); ++m3) {
                                    vectorized_matrix.push_back(cur_block(k3, m3));
                                }
                        }

    std::string filename_matrix{"/home/evgen/Education/MasterDegree/thesis/ED-researh/research/matrix_for_arseny/"
                                "4_4_4_4_4_4_192_192_k_1_e_1.bin"};
    save_complex_vector_binary(vectorized_matrix.begin(), vectorized_matrix.end(),filename_matrix);

    std::string filename_rhs{"/home/evgen/Education/MasterDegree/thesis/ED-researh/research/matrix_for_arseny/4_4_4_4_4_4_192_192_k_1_e_1_rhs.bin"};
    save_complex_vector_binary(b.begin(), b.end(),filename_rhs);

    std::string filename_sol{"/home/evgen/Education/MasterDegree/thesis/ED-researh/research/matrix_for_arseny/4_4_4_4_4_4_192_192_k_1_e_1_sol.bin"};
    save_complex_vector_binary(solution.begin(),solution.end(), filename_sol);
};
