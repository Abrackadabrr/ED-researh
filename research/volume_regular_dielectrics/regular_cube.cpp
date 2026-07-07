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

#include "math/fields/Utils.hpp"
#include "math/fourier/TripleToeplitz3x3Fourier.hpp"
#include "math/integration/decart/Integration.hpp"

#include "Utils.hpp"

#include <iostream>

using namespace EMW;

constexpr Types::scalar CUBE_LENGTH = 1.;
constexpr Types::scalar EPSILON = 4.;

Types::scalar homo_cube(const Types::point_t &x) { return EPSILON; }

#define MATRIX_COMPARISON 0
#define SYSTEM_SOLVING 1
#define CALC_RSP 1
#define CALC_FIELD 0

int main() {
    Eigen::setNbThreads(1);
    openblas_set_num_threads(1);
    // 1. Рисуем сетку
    constexpr Types::scalar cube_length = CUBE_LENGTH;
    for (Types::index Nx = 101; Nx < 131; Nx+=10) {
        // constexpr Types::index Nx = 61;
        const Types::index Ny = Nx;
        const Types::index Nz = Nx;
        constexpr auto start_point = Types::point_t{-cube_length / 2, -cube_length / 2, -cube_length / 2};
        constexpr auto end_point = Types::point_t{cube_length / 2, cube_length / 2, cube_length / 2};
        Mesh::VolumeMesh::CubeMeshWithData mesh{start_point, cube_length, cube_length, cube_length, Nx, Ny, Nz};
        mesh.setName("cube_" + std::to_string(Nx));

        const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();
        std::cout << cube_measure << std::endl;
        const Types::scalar basis_fn_module = 1. / sqrt(cube_measure);

        // Настраиваем диэлектрическую проницаемость
        mesh.smoothScalarData<DecartIntegration::NewtonCotess::Quadrature<1, 1, 1>>("eps", homo_cube);

        // 2. Параметры падающей волны
        constexpr Types::scalar freq = 0.3; // GHz
        constexpr Types::complex_d k{Physics::get_k_on_frquency(freq), 0.0};

        Physics::planeWaveCase incident_field{Types::Vector3d{1, 0, 0}, {k.real(), 0.}, Types::Vector3d{0, 0, 1}};
        std::cout << "Длина волны в свободном пространстве = " << 2 * M_PI / k.real() << std::endl;
        std::cout << "lambda_0 / mesh.h = " << 2 * M_PI / (k.real() * mesh.dx()) << std::endl;
        std::cout << "lambda / mesh.h = " << 2 * M_PI / (EPSILON * k.real() * mesh.dx()) << std::endl;

        // 3. Галеркинская проекция правой части
        Operators::Volume::ProjectorOnMesh proj{mesh};
        auto rhs = proj([incident_field](Types::point_t p) { return incident_field.value(p); });
        // Поправляем правую часть
        Types::VectorXc b = rhs * basis_fn_module;

        // 4. Галеркинская проекция оператора
        Operators::Volume::operator_K_over_cube_mesh operator_K{k, mesh};
        operator_K.set_tolerances(1e-7, 1e-22);
        operator_K.set_adaptive_integration_max_levels({20, 20, 10, 7});
        auto matrix = operator_K.compute_galerkin_matrix(basis_fn_module);
        std::cout << "Matrix sizes: " << matrix.cols() << ' ' << matrix.rows() << std::endl;


#if SYSTEM_SOLVING
        using fourier_t = Math::Fourier::TripleToeplitz3x3FourierParallel<Types::complex_d>;

        auto fourier = fourier_t(matrix);

        // Собираем матрицу (eps - 1) как вектор из значений
        Types::VectorXc diag_eps = Types::VectorXc::Zero(3 * mesh.getCells().size());
        const auto eps_data = mesh.getScalarData("eps");
        for (size_t idx = 0; idx < mesh.getCells().size(); ++idx) {
            diag_eps[3 * idx] = eps_data[idx] - 1.;
            diag_eps[3 * idx + 1] = eps_data[idx] - 1.;
            diag_eps[3 * idx + 2] = eps_data[idx] - 1.;
        }

        Math::LinAgl::Matrix::Wrappers::SimpleVolumeOperator A_compressed{fourier, diag_eps};

        // 6. Решаем системы
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

#pragma omp parallel for num_threads(14)
        for (auto &&[vvalue, hvalue, phi] : std::views::zip(rsp_vv, rsp_hh, phis)) {
            vvalue = (10 * std::log10(ESA::calculateRSP_kahan(get_tau_vv(phi), {k.real(), 0.}, "solution", mesh)));
            hvalue = (10 * std::log10(ESA::calculateRSP_kahan(get_tau_hh(phi), {k.real(), 0.}, "solution", mesh)));
        }

        auto degree_view = view | std::views::transform([&](Types::scalar phi) { return phi * 180 / M_PI; });
        std::vector<Types::scalar> phis_degree{degree_view.begin(), degree_view.end()};
        std::ofstream rsp_vv_file{path + "cube_sigma_vv_" + std::to_string(Nx) + ".csv"};
        std::ofstream rsp_hh_file{path + "cube_sigma_hh_" + std::to_string(Nx) + ".csv"};
        Utils::to_csv(phis_degree, rsp_vv, "angle", "rsp", rsp_vv_file);
        Utils::to_csv(phis_degree, rsp_hh, "angle", "rsp", rsp_hh_file);
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
