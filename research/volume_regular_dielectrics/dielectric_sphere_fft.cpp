//
// Created by evgen on 22.03.2026.
//

#include "EMW/types/Types.hpp"

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

#include "Utils.hpp"

#include <iostream>

using namespace EMW;

constexpr Types::scalar SPHERE_RADUIS = 0.5;
constexpr Types::scalar CUBE_LENGTH = 0.5;

Types::scalar homo_sphere(const Types::point_t &x) { return x.norm() < SPHERE_RADUIS ? 2.56 : 1; }

int main() {
    Eigen::setNbThreads(1);
    openblas_set_num_threads(1);
    // 1. Рисуем сетку
    constexpr Types::scalar cube_length = 1;
    constexpr Types::index Nx = 61;
    constexpr Types::index Ny = 61;
    constexpr Types::index Nz = 61;
    constexpr Types::scalar mesh_one_axis_size = cube_length / (Nx - 1);
    Mesh::VolumeMesh::CubeMeshWithData mesh{Types::point_t{-cube_length / 2, -cube_length / 2, -cube_length / 2},
                                            (Nx - 1) * mesh_one_axis_size,
                                            (Ny - 1) * mesh_one_axis_size,
                                            (Nz - 1) * mesh_one_axis_size,
                                            Nx,
                                            Ny,
                                            Nz};
    mesh.setName("sphere_" + std::to_string(Nx));
    const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();
    const Types::scalar basis_fn_module = 1. / sqrt(cube_measure);
    // Настраиваем диэлектрическую проницаемость
    mesh.smoothScalarData<DecartIntegration::GaussLegendre::Quadrature<2, 2, 2>>("eps", homo_sphere);

    // 2. Параметры падающей волны
    constexpr Types::scalar freq = 1; // GHz
    constexpr Types::complex_d k{Physics::get_k_on_frquency(freq), 0.};

    Physics::planeWaveCase incident_field{Types::Vector3d{0, 1, 0}, k, Types::Vector3d{1, 0, 0}};
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
    auto matrix = operator_K.compute_galerkin_matrix(basis_fn_module);
#if 1
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

    Math::LinAgl::Matrix::Wrappers::VolumeOperatorMatrixReplacement A_compressed{fourier, diag_eps};

    // 6. Решаем системы
    // Поправляем правую часть по маске из фиктивных элементов
    b = A_compressed.modify_rhs_according_to_mask(b);
    auto solution = Research::solve<Eigen::GMRES>(A_compressed, b, 2000, 2e-4);

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

    // 8. Расчситываем диаграмму направленности
    int N = 180;
    const auto get_tau_hh = [](Types::scalar phi) { return Types::Vector3d{std::cos(phi), 0, std::sin(phi)}; };
    const auto get_tau_vv = [](Types::scalar phi) { return Types::Vector3d{std::cos(phi), std::sin(phi), 0}; };

    auto view = std::views::iota(0, N) | std::views::transform([N](int i) { return i * M_PI / N; });
    std::vector<Types::scalar> phis{view.begin(), view.end()};

    std::vector<Types::scalar> rsp_hh{};
    rsp_hh.resize(N);
    std::vector<Types::scalar> rsp_vv{};
    rsp_vv.resize(N);

#pragma omp parallel for num_threads(14)
    for (auto &&[vvalue, hvalue, phi] : std::views::zip(rsp_vv, rsp_hh, phis)) {
        vvalue = (std::log(ESA::calculateRSP_kahan(get_tau_vv(phi), k, "solution", mesh)));
        hvalue = (std::log(ESA::calculateRSP_kahan(get_tau_hh(phi), k, "solution", mesh)));
    }

    auto degree_view = view | std::views::transform([&](Types::scalar phi) { return phi * 180 / M_PI; });
    std::vector<Types::scalar> phis_degree{degree_view.begin(), degree_view.end()};
    std::ofstream rsp_vv_file{path + "sigma_vv_" + std::to_string(Nx) + ".csv"};
    std::ofstream rsp_hh_file{path + "sigma_hh_" + std::to_string(Nx) + ".csv"};
    Utils::to_csv(phis_degree, rsp_vv, "angle", "rsp", rsp_vv_file);
    Utils::to_csv(phis_degree, rsp_hh, "angle", "rsp", rsp_hh_file);
#endif
};
