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

#include "nmie.hpp"

#include "Utils.hpp"

#include <complex>
#include <fstream>
#include <iostream>
#include <vector>

using namespace EMW;

constexpr Types::scalar SPHERE_RADUIS = 0.5;
constexpr Types::scalar CUBE_LENGTH = 0.5;
constexpr Types::scalar SPHERE_EPSILON = 2.56;

Types::scalar homo_sphere(const Types::point_t &x) { return x.norm() < SPHERE_RADUIS ? SPHERE_EPSILON : 1; }

#define MATRIX_COMPARISON 0
#define SYSTEM_SOLVING 1
#define CALC_RSP 1
#define CALC_FIELD 0
#define ANALYTICAL_CHECK 1

std::vector<Types::Vector3c> calculate_analytical_solution(const Mesh::VolumeMesh::CubeMeshWithData &mesh,
                                                           Types::scalar sphere_radius, Types::complex_d epsilon,
                                                           Types::complex_d wave_number) {
    const auto &cells = mesh.getCells();
    const size_t total_points = cells.size();
    const Types::scalar k0 = wave_number.real();

    // scattnlay works with dimensionless coordinates k0 * r and size parameter k0 * R.
    std::vector<double> layer_size{k0 * sphere_radius};
    // nField expects the refractive index of each layer, not the dielectric permittivity.
    std::vector<std::complex<double>> refractive_index{std::sqrt(epsilon)};
    std::vector<double> Xp(total_points);
    std::vector<double> Yp(total_points);
    std::vector<double> Zp(total_points);

    // The canonical incident wave in scattnlay is E_inc = x_hat * exp(i * k0 * z):
    // the wave vector is directed along +z, and the electric polarization is along +x.
    // Therefore no coordinate rotation is applied here.
    for (size_t idx = 0; idx < total_points; ++idx) {
        const auto &center = cells[idx].center_;
        Xp[idx] = k0 * center.x();
        Yp[idx] = k0 * center.y();
        Zp[idx] = k0 * center.z();
    }

    std::vector<std::vector<std::complex<double>>> E(total_points, std::vector<std::complex<double>>(3));
    std::vector<std::vector<std::complex<double>>> H(total_points, std::vector<std::complex<double>>(3));

    // nField returns the full analytical field in Cartesian components: incident + scattered.
    const int nmax = nmie::nField(1, -1, layer_size, refractive_index, -1, nmie::Modes::kAll, nmie::Modes::kAll,
                                  static_cast<unsigned int>(total_points), Xp, Yp, Zp, E, H);
    std::cout << "Analytical solution nmax = " << nmax << std::endl;

    std::vector<Types::Vector3c> field_on_mesh;
    field_on_mesh.reserve(total_points);
    for (size_t idx = 0; idx < total_points; ++idx) {
        if (cells[idx].center_.norm() < SPHERE_RADUIS)
            field_on_mesh.emplace_back(E[idx][0], E[idx][1], E[idx][2]);
        else
            field_on_mesh.emplace_back(Types::Vector3c::Zero());
    }

    return field_on_mesh;
}

struct MieRSP {
    std::vector<Types::scalar> hh;
    std::vector<Types::scalar> vv;
};

MieRSP calculate_mie_rsp(const std::vector<Types::scalar> &phis, Types::scalar sphere_radius,
                         Types::complex_d epsilon, Types::complex_d wave_number) {
    const Types::scalar k0 = wave_number.real();

    std::vector<double> layer_size{k0 * sphere_radius};
    std::vector<std::complex<double>> refractive_index{std::sqrt(epsilon)};
    std::vector<double> theta;
    theta.reserve(phis.size());
    for (const auto phi : phis) {
        theta.push_back(phi);
    }

    double Qext = 0;
    double Qsca = 0;
    double Qabs = 0;
    double Qbk = 0;
    double Qpr = 0;
    double g = 0;
    double Albedo = 0;
    std::vector<std::complex<double>> S1;
    std::vector<std::complex<double>> S2;

    const int nmax = nmie::nMie(1, layer_size, refractive_index, static_cast<unsigned int>(theta.size()), theta, &Qext,
                                &Qsca, &Qabs, &Qbk, &Qpr, &g, &Albedo, S1, S2);
    std::cout << "Mie RSP nmax = " << nmax << std::endl;

    MieRSP rsp;
    rsp.hh.resize(phis.size());
    rsp.vv.resize(phis.size());
    const Types::scalar scale = 4.0 * M_PI / (k0 * k0);
    for (size_t idx = 0; idx < phis.size(); ++idx) {
        rsp.vv[idx] = scale * std::norm(S1[idx]);
        rsp.hh[idx] = scale * std::norm(S2[idx]);
    }

    return rsp;
}

int main() {
    Eigen::setNbThreads(1);
    openblas_set_num_threads(1);
    // 1. Рисуем сетку
    constexpr Types::scalar cube_length = 2 * SPHERE_RADUIS;
    constexpr Types::index Nx_start = 11;
    constexpr Types::index Nx_end = 12;
    for (Types::index Nx = Nx_start; Nx < Nx_end; Nx+=20) {
        const Types::index Ny = Nx;
        const Types::index Nz = Nx;
        const Types::scalar mesh_one_axis_size = cube_length / (Nx - 1);
        Mesh::VolumeMesh::CubeMeshWithData mesh{Types::point_t{-cube_length / 2, -cube_length / 2, -cube_length / 2},
                                                (Nx - 1) * mesh_one_axis_size,
                                                (Ny - 1) * mesh_one_axis_size,
                                                (Nz - 1) * mesh_one_axis_size,
                                                Nx,
                                                Ny,
                                                Nz};
        mesh.setName("sphere_" + std::to_string(Nx));
        const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();
        std::cout << cube_measure << std::endl;
        const Types::scalar basis_fn_module = 1. / sqrt(cube_measure);
        // Настраиваем диэлектрическую проницаемость
        mesh.smoothScalarData<DecartIntegration::NewtonCotess::Quadrature<4, 4, 4>>("eps", homo_sphere);

        // 2. Параметры падающей волны
        constexpr Types::scalar freq = 1.; // GHz
        constexpr Types::complex_d k{Physics::get_k_on_frquency(freq), 0.0};

        Physics::planeWaveCase incident_field{Types::Vector3d{1, 0, 0}, {k.real(), 0.}, Types::Vector3d{0, 0, -1}};
        std::cout << "Длина волны в свободном пространстве = " << 2 * M_PI / k.real() << std::endl;
        std::cout << "lambda_0 / mesh.h = " << 2 * M_PI / (k.real() * mesh_one_axis_size) << std::endl;
        std::cout << "lambda / mesh.h = " << 2 * M_PI / (SPHERE_EPSILON * k.real() * mesh_one_axis_size) << std::endl;

        // 3. Галеркинская проекция правой части
        Operators::Volume::ProjectorOnMesh proj{mesh};
        auto rhs = proj([incident_field](Types::point_t p) { return incident_field.value(p); });
        // Поправляем правую часть
        Types::VectorXc b = rhs * basis_fn_module;

        // 4. Галеркинская проекция оператора
        Operators::Volume::operator_K_over_cube_mesh operator_K{k, mesh};
        operator_K.set_tolerances(1e-6, 1e-21);
        operator_K.set_adaptive_integration_max_levels({10, 10, 10, 10});
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

#if ANALYTICAL_CHECK
        // Расчет аналитического решения на сфере
        auto analytical_solution = calculate_analytical_solution(
            mesh, SPHERE_RADUIS, Types::complex_d{SPHERE_EPSILON, 0}, k);
        mesh.setVectorData("analytical_solution", std::move(analytical_solution));
        const auto an_sol = mesh.getVectorDataAsVector("analytical_solution");
        //  и расчет невязки по построенной матрице
        const auto residual = (an_sol - fourier.matvec(an_sol.cwiseProduct(diag_eps))).cwiseProduct(mask) / basis_fn_module - b.cwiseProduct(mask);
        std::cout << residual.norm() / b.norm() << std::endl;
#endif

#if MATRIX_COMPARISON
        operator_K.set_adaptive_integration_max_levels({80, 40, 40, 20});
        auto matrix_2 = operator_K.compute_galerkin_matrix(basis_fn_module);

        auto rel_err = Utils::relative_frobenius_error(matrix, matrix_2);
        std::cout << "Relative error in matrixies = " << rel_err << std::endl;
#endif

#if SYSTEM_SOLVING
        Math::LinAgl::Matrix::Wrappers::VolumeOperatorMatrixReplacement A_compressed{fourier, diag_eps};

        // 6. Решаем системы
        // Поправляем правую часть по маске из фиктивных элементов
        b = A_compressed.modify_rhs_according_to_mask(b);
        auto solution = Research::solve<Eigen::GMRES>(A_compressed, b, 10000, 1e-5);

        // 7. Преобразовываем в векторное поле на ячейках и пишем в данные сетки
        std::vector<Types::Vector3c> field_on_mesh;
        field_on_mesh.reserve(mesh.getCells().size());
        for (size_t idx = 0; idx < mesh.getCells().size(); ++idx) {
            field_on_mesh.emplace_back(solution[3 * idx + 0] * basis_fn_module, solution[3 * idx + 1] * basis_fn_module,
                                       solution[3 * idx + 2] * basis_fn_module);
        }

        // Добавляем поле
        mesh.setVectorData("solution", std::move(field_on_mesh));

        // Отрисовываем результат
        const std::string path =
            "/home/evgen/Education/MasterDegree/thesis/ED-researh/research/volume_regular_dielectrics/";
        VTK::volume_mesh_withdata_snapshot(mesh, path);

#if ANALYTICAL_CHECK
        const Types::scalar phase_shift = std::arg(an_sol[3 * Nx * Ny * Nz / 2]) - std::arg(solution[3 * Nx * Ny * Nz / 2]);
        std::cout << "Phase shift: " << phase_shift << std::endl;
        const Types::VectorXc solution_difference = std::exp(Math::Constants::i * phase_shift) * solution - an_sol.cwiseProduct(mask) / basis_fn_module;
        std::cout << "solution difference relative norm = " << solution_difference.norm() / solution.norm() << std::endl;
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

        auto mie_rsp = calculate_mie_rsp(phis, SPHERE_RADUIS, Types::complex_d{SPHERE_EPSILON, 0}, k);
        for (auto &value : mie_rsp.vv) {
            value = 10 * std::log10(value);
        }
        for (auto &value : mie_rsp.hh) {
            value = 10 * std::log10(value);
        }

#pragma omp parallel for num_threads(14)
        for (auto &&[vvalue, hvalue, an_vvalue, an_hvalue, phi] :
             std::views::zip(rsp_vv, rsp_hh, an_rsp_vv, an_rsp_hh, phis)) {
            vvalue = (10 * std::log10(ESA::calculateRSP_kahan(get_tau_vv(phi), {k.real(), 0.}, "solution", mesh)));
            hvalue = (10 * std::log10(ESA::calculateRSP_kahan(get_tau_hh(phi), {k.real(), 0.}, "solution", mesh)));
            an_vvalue =
                (10 * std::log10(ESA::calculateRSP_kahan(get_tau_vv(phi), {k.real(), 0.}, "analytical_solution", mesh)));
            an_hvalue =
                (10 * std::log10(ESA::calculateRSP_kahan(get_tau_hh(phi), {k.real(), 0.}, "analytical_solution", mesh)));
             }

        auto degree_view = view | std::views::transform([&](Types::scalar phi) { return phi * 180 / M_PI; });
        std::vector<Types::scalar> phis_degree{degree_view.begin(), degree_view.end()};
        std::ofstream rsp_vv_file{path + "sphere_sigma_vv_" + std::to_string(Nx) + ".csv"};
        std::ofstream rsp_hh_file{path + "sphere_sigma_hh_" + std::to_string(Nx) + ".csv"};
        std::ofstream an_rsp_vv_file{path + "an_sphere_sigma_vv_" + std::to_string(Nx) + ".csv"};
        std::ofstream an_rsp_hh_file{path + "an_sphere_sigma_hh_" + std::to_string(Nx) + ".csv"};
        std::ofstream mie_rsp_vv_file{path + "mie_sigma_vv_" + std::to_string(Nx) + ".csv"};
        std::ofstream mie_rsp_hh_file{path + "mie_sigma_hh_" + std::to_string(Nx) + ".csv"};
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

#pragma omp parallel for num_threads(14)
        for (size_t idx = 0; idx < meshgrid.size(); ++idx) {
            field[idx] = operator_K.compute_arbitrary_point(meshgrid[idx], mesh.getVectorData("solution"));
        }

        VTK::field_in_points_snapshot({field}, {}, {"E"}, {}, meshgrid, "electric_filed_" + std::to_string(Nx) , path);
#endif
    }
};
