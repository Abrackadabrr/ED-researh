//
// Created by evgen on 10.07.2026.
//

#include "EMW/types/Types.hpp"

#include "mesh/Utils.hpp"
#include "mesh/volume_mesh/CubeMeshWithData.hpp"

#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "experiment/ESA.hpp"
#include "experiment/PhysicalCondition.hpp"

#include "../Solve.hpp"

#include "MatrixReplacement.hpp"
#include "MatrixTraits.hpp"
#include "Preconditioning.hpp"
#include "analytical_solution/SphereMieAnalytical.hpp"

#include "math/fields/Utils.hpp"
#include "math/fourier/TripleToeplitz3x3Fourier.hpp"
#include "math/integration/decart/Integration.hpp"

#include "Utils.hpp"

#include <fstream>
#include <iostream>
#include <vector>

using namespace EMW;

// 4/3 pi r^3 = V => r = cbrt(3 V / 4 pi)

Types::scalar sphere_effective_radius(const Types::VectorXc& epsilon_data, Types::scalar volume_of_cube) {
    Types::VectorXd mask = Types::VectorXd::Zero(epsilon_data.size());
    for (Types::index i = 0; i < epsilon_data.size(); ++i) {
        mask[i] = (std::abs(epsilon_data[i] - 1.) > 1e-10);
    }
    return std::pow(3 * mask.sum() * volume_of_cube / (4 * M_PI), 1./3.);
};

int main() {
    Eigen::setNbThreads(1);
    // характеритики сферы
    constexpr Types::complex_d SPHERE_EPSILON = {2.56, 0.};
    constexpr Types::scalar SPHERE_RADUIS = 0.5;
    const auto homo_sphere = [SPHERE_RADUIS, SPHERE_EPSILON](const Types::point_t &x) {
        return x.norm() < SPHERE_RADUIS ? SPHERE_EPSILON : Types::complex_d{1, 0};
    };
    Types::scalar effective_sphere_radius; // посчитается позже
    // параметры сетки
    constexpr Types::scalar cube_length = 2 * SPHERE_RADUIS;
    constexpr Types::index Nx_start = 41;
    constexpr Types::index Nx_end = 42;
    // настройки для расчета оператора
    constexpr Types::scalar rTol = 1e-6;
    constexpr Types::scalar aTol = 1e-21;
    constexpr Types::index lev_2d = 3;
    constexpr Types::index lev_3d = 3;
    constexpr Types::index lev_4d = 2;
    constexpr Types::index lev_6d = 2;
    constexpr Types::index nearness_trh = 2;
    // параметры падающего излучения
    constexpr Types::scalar freq = 0.695; // GHz
    constexpr Types::complex_d k = Physics::get_k_on_frquency(freq) * Types::complex_d{1.0, 0.};
    constexpr Types::Vector3d polarization{1, 0, 0};
    constexpr Types::Vector3d k_vector{0, 0, -1};
    // путь для сохранения результатов
    const std::string path = "/home/evgen/Education/MasterDegree/thesis/ED-researh/research/cluster_dielectric/";

    for (Types::index Nx = Nx_start; Nx < Nx_end; Nx += 20) {
        // 1. Рисуем сетку
        const Types::index Ny = Nx;
        const Types::index Nz = Nx;
        Mesh::VolumeMesh::CubeMeshWithData mesh{Types::point_t{-cube_length / 2, -cube_length / 2, -cube_length / 2},
                                                cube_length,
                                                cube_length,
                                                cube_length,
                                                Nx,
                                                Ny,
                                                Nz};

        mesh.setName("sphere_" + std::to_string(Nx));

        std::cout << Nx - 1 << " cubes per size" << std::endl;

        // Настраиваем диэлектрическую проницаемость
        mesh.smoothScalarData<DecartIntegration::NewtonCotess::Quadrature<1, 1, 1>>("eps", homo_sphere);

        const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();

        effective_sphere_radius = sphere_effective_radius(mesh.getScalarDataAsVector("eps"), cube_measure);

        std::cout.precision(16);
        std::cout << "Effective shpere raduis appeared to be\t " << effective_sphere_radius << std::endl;
        std::cout << "Real shpere raduis is\t\t\t\t\t " << SPHERE_RADUIS << std::endl;

        // 2. Параметры падающей волны
        Physics::planeWaveCase incident_field{polarization, {k.real(), 0.}, k_vector};
        std::cout << "Длина волны в свободном пространстве = " << 2 * M_PI / k.real() << std::endl;
        std::cout << "lambda_0 / mesh.h = " << 2 * M_PI / (k.real() * cube_length / (Nx - 1)) << std::endl;
        std::cout << "lambda / mesh.h = " << 2 * M_PI / (std::sqrt(std::abs(SPHERE_EPSILON)) * k.real() * cube_length / (Nx - 1))
                  << std::endl;

        // 3. Галеркинская проекция правой части
        const Types::scalar basis_fn_module = 1. / sqrt(cube_measure);

        Operators::Volume::ProjectorOnMesh proj{mesh};
        auto rhs = proj([incident_field](Types::point_t p) { return incident_field.value(p); });
        // Поправляем правую часть
        Types::VectorXc b = rhs * basis_fn_module;

        // 4. Галеркинская проекция оператора
        Operators::Volume::operator_K_over_cube_mesh operator_K{k, mesh};
        operator_K.set_tolerances(rTol, aTol);
        operator_K.set_adaptive_integration_max_levels({lev_2d, lev_3d, lev_4d, lev_6d});
        operator_K.set_nearness_threshold(nearness_trh);
        auto matrix = operator_K.compute_galerkin_matrix(basis_fn_module);
        std::cout << "Matrix sizes: " << matrix.cols() << " х " << matrix.rows() << std::endl;

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

        // 5. Расчет аналитического решения на сфере
        auto analytical_solution = Research::VolDie::Mie::calculate_field_on_mesh(
            mesh, 0.5, SPHERE_EPSILON, {k.real(), 0.});
        mesh.setVectorData("analytical_solution", std::move(analytical_solution));

        // 6. Решаем системы
        Math::LinAgl::Matrix::Wrappers::VolumeOperatorMatrixReplacement A_compressed{fourier, diag_eps};
        // Поправляем правую часть по маске из фиктивных элементов
        b = A_compressed.modify_rhs_according_to_mask(b);
        auto solution = Research::solve<Eigen::GMRES>(A_compressed, b, 10000, 1e-9, 500);

        // 7. Преобразовываем в векторное поле на ячейках и пишем в данные сетки
        std::vector<Types::Vector3c> field_on_mesh;
        field_on_mesh.reserve(mesh.getCells().size());
        for (size_t idx = 0; idx < mesh.getCells().size(); ++idx) {
            field_on_mesh.emplace_back(solution[3 * idx + 0] * basis_fn_module, solution[3 * idx + 1] * basis_fn_module,
                                       solution[3 * idx + 2] * basis_fn_module);
        }

        // Добавляем поле
        mesh.setVectorData("solution", std::move(field_on_mesh));

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

        // Сохраняем результаты
        auto degree_view = view | std::views::transform([&](Types::scalar phi) { return phi * 180 / M_PI; });
        std::vector<Types::scalar> phis_degree{degree_view.begin(), degree_view.end()};
        std::ofstream rsp_vv_file{path + "sphere_sigma_vv_" + std::to_string(Nx) + ".csv"};
        std::ofstream rsp_hh_file{path + "sphere_sigma_hh_" + std::to_string(Nx) + ".csv"};
        std::ofstream an_rsp_vv_file{path + "an_sphere_sigma_vv_" + std::to_string(Nx) + ".csv"};
        std::ofstream an_rsp_hh_file{path + "an_sphere_sigma_hh_" + std::to_string(Nx) + ".csv"};
        Utils::to_csv(phis_degree, rsp_vv, "angle", "rsp", rsp_vv_file);
        Utils::to_csv(phis_degree, rsp_hh, "angle", "rsp", rsp_hh_file);
        Utils::to_csv(phis_degree, an_rsp_vv, "angle", "rsp", an_rsp_vv_file);
        Utils::to_csv(phis_degree, an_rsp_hh, "angle", "rsp", an_rsp_hh_file);
    }

    // референсный теоретический расчет
    int N = 360;
    auto view = std::views::iota(0, N) | std::views::transform([N](int i) { return i * M_PI / N; });
    std::vector<Types::scalar> phis{view.begin(), view.end()};
    auto degree_view = view | std::views::transform([&](Types::scalar phi) { return phi * 180 / M_PI; });
    std::vector<Types::scalar> phis_degree{degree_view.begin(), degree_view.end()};

    auto mie_rsp = Research::VolDie::Mie::calculate_rsp(
        phis, 0.5, SPHERE_EPSILON, {k.real(), 0.});
    for (auto &value : mie_rsp.vv) {
        value = 10 * std::log10(value) + 20;
    }
    for (auto &value : mie_rsp.hh) {
        value = 10 * std::log10(value) + 20;
    }
    std::ofstream mie_rsp_vv_file{path + "mie_sigma_vv.csv"};
    std::ofstream mie_rsp_hh_file{path + "mie_sigma_hh.csv"};

    Utils::to_csv(phis_degree, mie_rsp.vv, "angle", "rsp", mie_rsp_vv_file);
    Utils::to_csv(phis_degree, mie_rsp.hh, "angle", "rsp", mie_rsp_hh_file);
};
