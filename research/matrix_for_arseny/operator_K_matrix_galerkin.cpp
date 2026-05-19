//
// Created by evgen on 22.03.2026.
//

#include "EMW/types/Types.hpp"

#include "mesh/volume_mesh/CubeMeshWithData.hpp"

#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "experiment/PhysicalCondition.hpp"

#include "math/fields/Utils.hpp"

#include "Utils.hpp"

#include <fstream>
#include <iostream>

using namespace EMW;

void save_complex_vector_binary(const std::vector<std::complex<double>> &data, const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);

    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    const auto size = static_cast<std::uint64_t>(data.size());
    out.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (const auto &z : data) {
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
    Eigen::setNbThreads(1);
    openblas_set_num_threads(1);
    // 1. Рисуем сетку
    constexpr Types::scalar cube_length = 1;
    constexpr Types::index Nx = 20;
    constexpr Types::index Ny = Nx;
    constexpr Types::index Nz = Nx;
    constexpr Types::index nx = 4;
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
    mesh.setName("sphere_20");
    const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();
    const Types::scalar basis_fn_module = 1. / sqrt(cube_measure);

    // 2. Параметры падающей волны
    constexpr Types::complex_d k{1, 0.};
    Physics::planeWaveCase incident_field{Types::Vector3d{0, 1, 0}, k, Types::Vector3d{1, 0, 0}};
    std::cout << "K real = " << k.real() << std::endl;
    std::cout << "Длина волны в свободном пространстве = " << 2 * M_PI / k.real() << std::endl;
    std::cout << "lambda_0 / mesh.h = " << 2 * M_PI / (k.real() * mesh_one_axis_size) << std::endl;
    std::cout << "lambda / mesh.h = " << 2 * M_PI / (4 * k.real() * mesh_one_axis_size) << std::endl;

    // 3. Галеркинская проекция правой части
    Operators::Volume::ProjectorOnMesh proj{mesh};
    auto rhs = proj([incident_field](Types::point_t p) { return incident_field.value(p); });
    // Поправляем правую часть
    Types::VectorXc b = rhs * basis_fn_module;

    // 4. Галеркинская проекция оператора
    Operators::Volume::operator_K_over_cube_mesh operator_K{k, mesh};
    auto matrix = operator_K.compute_galerkin_matrix_custom_blocksize(nx, ny, nz, basis_fn_module).matrix;

    std::cout << "matrix assembled" << std::endl;

    // Сборка в виде вектора с правильной индексацией
    std::vector<Types::complex_d> vectorized_matrix{};
    vectorized_matrix.reserve(matrix.rows() * matrix.cols());
    size_t counter = 0;
    for (size_t i3 = 0; i3 < Nz / nz; ++i3)
        for (size_t j3 = 0; j3 < Nz / nz; ++j3)
            for (size_t i2 = 0; i2 < Ny / ny; ++i2)
                for (size_t j2 = 0; j2 < Ny / ny; ++j2)
                    for (size_t i1 = 0; i1 < Nx / nx; ++i1)
                        for (size_t j1 = 0; j1 < Nx / nx; ++j1) {
                            auto cur_block = matrix.get_block(i3, j3).get_block(i2, j2).get_block(i1, j1);
                            for (size_t k3 = 0; k3 < nz; ++k3)
                                for (size_t m3 = 0; m3 < nz; ++m3) {
                                    size_t row_index_part_3 = k3 * nx * ny;
                                    size_t col_index_part_3 = m3 * nx * ny;

                                    for (size_t k2 = 0; k2 < ny; ++k2)
                                        for (size_t m2 = 0; m2 < ny; ++m2) {
                                            size_t row_index_part_2 = k2 * nx;
                                            size_t col_index_part_2 = m2 * nx;

                                            for (size_t k1 = 0; k1 < nx; ++k1)
                                                for (size_t m1 = 0; m1 < nx; ++m1) {
                                                    size_t row_index_part_1 = k1 * 1;
                                                    size_t col_index_part_1 = m1 * 1;

                                                    size_t bottom_matrix_index_row =
                                                        row_index_part_1 + row_index_part_2 + row_index_part_3;
                                                    size_t bottom_matrix_index_col =
                                                        col_index_part_1 + col_index_part_2 + col_index_part_3;

                                                    auto bottom_matrix_block = cur_block.block(
                                                        bottom_matrix_index_row, bottom_matrix_index_col, 3, 3);
                                                    for (size_t p = 0; p < 3; ++p)
                                                        for (size_t q = 0; q < 3; ++q) {
                                                            vectorized_matrix.push_back(bottom_matrix_block(p, q));
                                                            counter++;
                                                        }
                                                }
                                        }
                                }
                        }
    std::cout << matrix.rows() * matrix.cols() << std::endl;
    std::cout << counter << std::endl;
    // 6. Дамп в файл
    std::string filename{"/home/evgen/Education/MasterDegree/thesis/ED-researh/research/matrix_for_arseny/5_5_5_4_4_4_3_k_eq_1.bin"};

    save_complex_vector_binary(vectorized_matrix,filename);

};
