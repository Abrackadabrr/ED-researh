//
// Created by evgen on 22.03.2026.
//

#include "EMW/types/Types.hpp"

#include "mesh/Utils.hpp"
#include "mesh/volume_mesh/CubeMeshWithData.hpp"

#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "experiment/ESA.hpp"
#include "experiment/PhysicalCondition.hpp"

#include "../Solve.hpp"

#include "visualisation/include/VTKFunctions.hpp"

#include "MatrixReplacement.hpp"
#include "MatrixTraits.hpp"
#include "Preconditioning.hpp"

#include "math/fields/Utils.hpp"
#include "math/fourier/TripleToeplitz3x3Fourier.hpp"
#include "math/integration/decart/Integration.hpp"

#include "../cluster_dielectric/analytical_solution/SphereMieAnalytical.hpp"
#include "Utils.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace EMW;

constexpr Types::scalar SPHERE_RADUIS = 0.5;
constexpr Types::scalar CUBE_LENGTH = 0.5;
constexpr Types::scalar SPHERE_EPSILON = 2.56;

Types::scalar homo_sphere(const Types::point_t &x) { return x.norm() < SPHERE_RADUIS ? SPHERE_EPSILON : 1; }

#define MATRIX_COMPARISON 1
#define SYSTEM_SOLVING 1
#define CALC_RSP 1
#define CALC_FIELD 0
#define ANALYTICAL_CHECK 1

int main() {
    Eigen::setNbThreads(1);
    const std::string output_path =
        "/home/evgen/Education/MasterDegree/thesis/ED-researh/research/volume_regular_dielectrics/";

#if SYSTEM_SOLVING && ANALYTICAL_CHECK
    std::ofstream convergence_file{output_path + "dielectric_sphere_fft_convergence.csv"};
    if (!convergence_file) {
        throw std::runtime_error("Failed to open dielectric-sphere convergence CSV file");
    }
    convergence_file << "cells_per_diameter,mie_discrete_residual,relative_l2_error,relative_c_error\n"
                     << std::setprecision(17);
#endif

    // 1. Рисуем сетку
    constexpr Types::scalar cube_length = 2 * SPHERE_RADUIS;
    constexpr Types::index Nx_start = 11;
    constexpr Types::index Nx_end = 32;
    for (Types::index Nx = Nx_start; Nx < Nx_end; Nx += 10) {
        const Types::index Ny = Nx;
        const Types::index Nz = Nx;
        const Types::scalar mesh_one_axis_size = cube_length / (Nx - 1);
        const auto minCorner = Types::point_t{-cube_length / 2, -cube_length / 2, -cube_length / 2};
        Mesh::VolumeMesh::CubeMeshWithData mesh{minCorner, cube_length, cube_length, cube_length, Nx, Ny, Nz};

        mesh.setName("sphere_" + std::to_string(Nx));
        const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();
        std::cout << cube_measure << std::endl;
        const Types::scalar basis_fn_module = 1. / sqrt(cube_measure);
        // Настраиваем диэлектрическую проницаемость
        mesh.smoothScalarData<DecartIntegration::NewtonCotess::Quadrature<1, 1, 1>>("eps", homo_sphere);

        // 2. Параметры падающей волны
        constexpr Types::scalar freq = 0.3; // GHz
        constexpr Types::complex_d k{Physics::get_k_on_frquency(freq), 0.0};

        Physics::planeWaveCase incident_field{Types::Vector3d{1, 0, 0}, {k.real(), 0.}, Types::Vector3d{0, 0, -1}};
        std::cout << "Длина волны в свободном пространстве = " << 2 * M_PI / k.real() << std::endl;
        std::cout << "lambda_0 / mesh.h = " << 2 * M_PI / (k.real() * mesh_one_axis_size) << std::endl;
        std::cout << "lambda / mesh.h = " << 2 * M_PI / (std::sqrt(SPHERE_EPSILON) * k.real() * mesh_one_axis_size) << std::endl;

        // 3. Галеркинская проекция правой части
        Operators::Volume::ProjectorOnMesh proj{mesh};
        auto rhs = proj([incident_field](Types::point_t p) { return incident_field.value(p); });
        // Поправляем правую часть
        Types::VectorXc b = rhs * basis_fn_module;

        // 4. Галеркинская проекция оператора
        Operators::Volume::operator_K_over_cube_mesh operator_K{k, mesh};
        operator_K.set_tolerances(1e-6, 1e-21);
        operator_K.set_adaptive_integration_max_levels({7, 7, 5, 5});
        operator_K.set_nearness_threshold(2);
        omp_set_num_threads(16);
        auto matrix = operator_K.compute_galerkin_matrix(basis_fn_module);
        std::cout << "Matrix sizes: " << matrix.cols() << ' ' << matrix.rows() << std::endl;

        // Собираем матрицу для быстрой умножалки
        using fourier_t = Math::Fourier::TripleToeplitz3x3FourierParallel<Types::complex_d>;
        auto fourier = fourier_t(matrix);

        // Собираем матрицу (eps - 1) как вектор из значений
        Types::VectorXc diag_eps = Types::VectorXc::Zero(3 * mesh.getCells().size());
        Types::VectorXd mask = Types::VectorXd::Zero(3 * mesh.getCells().size());
        const auto eps_data = mesh.getScalarData("eps");
        for (size_t idx = 0; idx < mesh.getCells().size(); ++idx) {
            const auto eps_m_1 = eps_data[idx] - 1.;
            diag_eps[3 * idx] = eps_m_1;
            diag_eps[3 * idx + 1] = eps_m_1;
            diag_eps[3 * idx + 2] = eps_m_1;
            if (std::abs(eps_m_1) < 1e-14)
                mask.block(3 * idx, 0, 3, 1) = Types::Vector3d::Zero();
            else
                mask.block(3 * idx, 0, 3, 1) = Types::Vector3d::Ones();
        }

        // Расчет аналитического решения на сфере
        auto analytical_solution = Research::VolDie::Mie::calculate_field_on_mesh(
            mesh, SPHERE_RADUIS, Types::complex_d{SPHERE_EPSILON, 0}, k);
        mesh.setVectorData("analytical_solution", std::move(analytical_solution));
        const auto an_sol = mesh.getVectorDataAsVector("analytical_solution");

#if ANALYTICAL_CHECK
        //  Расчет невязки по построенной матрице
        const Types::VectorXc residual =
            ((an_sol - fourier.matvec(an_sol.cwiseProduct(diag_eps))).cwiseProduct(mask) /
                basis_fn_module -
            b.cwiseProduct(mask))
                .eval();
        const Types::scalar relative_mie_residual = residual.norm() / b.cwiseProduct(mask).norm();
        std::cout << relative_mie_residual << std::endl;
#endif

#if MATRIX_COMPARISON
        operator_K.set_adaptive_integration_max_levels({10, 10, 5, 5});
        auto matrix_2 = operator_K.compute_galerkin_matrix(basis_fn_module);

        auto rel_err = Utils::relative_frobenius_error(matrix, matrix_2);
        std::cout << "Relative error in matrixies = " << rel_err << std::endl;
#endif

#if SYSTEM_SOLVING
        Math::LinAgl::Matrix::Wrappers::VolumeOperatorMatrixReplacement A_compressed{fourier, diag_eps};

        // 6. Решаем системы
        // Поправляем правую часть по маске из фиктивных элементов
        b = A_compressed.modify_rhs_according_to_mask(b);
        auto solution = Research::solve<Eigen::GMRES>(A_compressed, b, 10000, 1e-5, 1000);

        // 7. Преобразовываем в векторное поле на ячейках и пишем в данные сетки
        std::vector<Types::Vector3c> field_on_mesh;
        std::vector<Types::Vector3c> solution_difference;
        field_on_mesh.reserve(mesh.getCells().size());
        solution_difference.reserve(mesh.getCells().size());
        for (size_t idx = 0; idx < mesh.getCells().size(); ++idx) {
            const Types::Vector3c numerical_field{solution[3 * idx] * basis_fn_module,
                                                  solution[3 * idx + 1] * basis_fn_module,
                                                  solution[3 * idx + 2] * basis_fn_module};
            const Types::Vector3c mie_field{an_sol[3 * idx], an_sol[3 * idx + 1], an_sol[3 * idx + 2]};
            field_on_mesh.emplace_back(numerical_field);
            solution_difference.emplace_back(numerical_field - mie_field);
        }

        // Добавляем численное поле и его разность с решением Ми
        mesh.setVectorData("solution", std::move(field_on_mesh));
        mesh.setVectorData("solution_difference", std::move(solution_difference));

        // Отрисовываем результат
        VTK::volume_mesh_withdata_snapshot(mesh, output_path);

#if ANALYTICAL_CHECK
        const auto comparison_errors = Research::VolDie::Mie::compare_solutions_in_sphere_interior(
            mesh, "solution", "analytical_solution", SPHERE_RADUIS);
        std::cout << "Number of fully interior cells = " << comparison_errors.compared_cells << '\n'
                  << "C-norm error = " << comparison_errors.c_norm << '\n'
                  << "Relative C-norm error = " << comparison_errors.relative_c_norm << '\n'
                  << "L2-norm error = " << comparison_errors.l2_norm << '\n'
                  << "Relative L2-norm error = " << comparison_errors.relative_l2_norm << std::endl;

        convergence_file << Nx - 1 << ',' << relative_mie_residual << ','
                         << comparison_errors.relative_l2_norm << ','
                         << comparison_errors.relative_c_norm << '\n';
        convergence_file.flush();
#endif

#endif

#if CALC_RSP
        // 8. Расчситываем диаграмму направленности
        int N = 360;
        const auto get_tau_hh = [](Types::scalar phi) { return Types::Vector3d{std::sin(phi), 0, std::cos(phi)}; };
        const auto get_tau_vv = [](Types::scalar phi) { return Types::Vector3d{0, std::sin(phi), std::cos(phi)}; };

        auto view = std::views::iota(0, N) | std::views::transform([N](int i) { return i * M_PI / N; });
        std::vector<Types::scalar> phis{view.begin(), view.end()};

        std::vector<Types::scalar> rsp_hh{};
        rsp_hh.resize(N);
        std::vector<Types::scalar> rsp_vv{};
        rsp_vv.resize(N);
        std::vector<Types::scalar> an_rsp_hh{};
        an_rsp_hh.resize(N);
        std::vector<Types::scalar> an_rsp_vv{};
        an_rsp_vv.resize(N);

        auto mie_rsp = Research::VolDie::Mie::calculate_rsp(
            phis, SPHERE_RADUIS, Types::complex_d{SPHERE_EPSILON, 0}, k);
        for (auto &value : mie_rsp.vv) {
            value = 10 * std::log10(value);
        }
        for (auto &value : mie_rsp.hh) {
            value = 10 * std::log10(value);
        }

#pragma omp parallel for
        for (auto &&[vvalue, hvalue, an_vvalue, an_hvalue, phi] :
             std::views::zip(rsp_vv, rsp_hh, an_rsp_vv, an_rsp_hh, phis)) {
            vvalue = (10 * std::log10(ESA::calculateRSP_kahan(get_tau_vv(phi), {k.real(), 0.}, "solution", mesh)));
            hvalue = (10 * std::log10(ESA::calculateRSP_kahan(get_tau_hh(phi), {k.real(), 0.}, "solution", mesh)));
            an_vvalue = (10 * std::log10(ESA::calculateRSP_kahan(get_tau_vv(phi), {k.real(), 0.}, "analytical_solution",
                                                                 mesh)));
            an_hvalue = (10 * std::log10(ESA::calculateRSP_kahan(get_tau_hh(phi), {k.real(), 0.}, "analytical_solution",
                                                                 mesh)));
        }

        auto degree_view = phis | std::views::transform([&](Types::scalar phi) { return phi * 180 / M_PI; });
        std::vector<Types::scalar> phis_degree{degree_view.begin(), degree_view.end()};

        std::cout << "RCS calculated" << std::endl;

        std::ofstream rsp_vv_file{output_path + "sphere_sigma_vv_" + std::to_string(Nx) + ".csv"};
        std::ofstream rsp_hh_file{output_path + "sphere_sigma_hh_" + std::to_string(Nx) + ".csv"};
        std::ofstream an_rsp_vv_file{output_path + "an_sphere_sigma_vv_" + std::to_string(Nx) + ".csv"};
        std::ofstream an_rsp_hh_file{output_path + "an_sphere_sigma_hh_" + std::to_string(Nx) + ".csv"};
        std::ofstream mie_rsp_vv_file{output_path + "mie_sigma_vv_" + std::to_string(Nx) + ".csv"};
        std::ofstream mie_rsp_hh_file{output_path + "mie_sigma_hh_" + std::to_string(Nx) + ".csv"};
        Utils::to_csv(phis_degree, rsp_vv, "angle", "rsp", rsp_vv_file);
        Utils::to_csv(phis_degree, rsp_hh, "angle", "rsp", rsp_hh_file);
        Utils::to_csv(phis_degree, an_rsp_vv, "angle", "rsp", an_rsp_vv_file);
        Utils::to_csv(phis_degree, an_rsp_hh, "angle", "rsp", an_rsp_hh_file);
        Utils::to_csv(phis_degree, mie_rsp.vv, "angle", "rsp", mie_rsp_vv_file);
        Utils::to_csv(phis_degree, mie_rsp.hh, "angle", "rsp", mie_rsp_hh_file);
#endif

#if CALC_FIELD
        // 9. Считаем поле
        const int N = 100;
        const Types::scalar h = 0.04;
        std::vector<Mesh::point_t> meshgrid;
        meshgrid.reserve(N * N);
        Mesh::Utils::cartesian_product_unevenXY(std::ranges::views::iota(0, N), std::ranges::views::iota(0, N),
                                                std::back_inserter(meshgrid), N, N, h, h);
        std::vector<Types::Vector3c> field; field.resize(meshgrid.size());

#pragma omp parallel for
        for (size_t idx = 0; idx < meshgrid.size(); ++idx) {
            field[idx] = operator_K.compute_arbitrary_point(meshgrid[idx], mesh.getVectorData("solution"));
        }

        VTK::field_in_points_snapshot({field}, {}, {"E"}, {}, meshgrid, "electric_filed_" + std::to_string(Nx),
                                      output_path);
#endif
    }
};
