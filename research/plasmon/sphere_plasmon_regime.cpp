// Scattering by a homogeneous sphere in the plasmon regime.

#include "EMW/types/Types.hpp"

#include "mesh/volume_mesh/CubeMeshWithData.hpp"

#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "experiment/ESA.hpp"
#include "experiment/PhysicalCondition.hpp"

#include "visualisation/include/VTKFunctions.hpp"

#include "../Solve.hpp"

#include "../cluster_dielectric/MatrixReplacement.hpp"
#include "../cluster_dielectric/MatrixTraits.hpp"
#include "../cluster_dielectric/analytical_solution/SphereMieAnalytical.hpp"

#include "math/fourier/TripleToeplitz3x3Fourier.hpp"
#include "math/integration/decart/Integration.hpp"

#include "EMW/Utils.hpp"
#include "configs/SpherePlasmonRegimeConfig.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace EMW;

namespace {

void ensure_output_stream_is_open(const std::ofstream &stream, const std::filesystem::path &path) {
    if (!stream) {
        throw std::runtime_error("Cannot open output file '" + path.string() + "'");
    }
}

} // namespace

int main(int argc, char **argv) {
    Eigen::setNbThreads(1);

    const auto config = Research::Plasmon::load_sphere_plasmon_regime_config(argc, argv);

    const auto homogeneous_sphere = [&config](const Types::point_t &point) {
        return point.norm() < config.sphere.radius ? config.sphere.epsilon : Types::complex_d{1.0, 0.0};
    };

    const Types::scalar additional_mie_radius = 0;

    const Types::scalar cube_length = 2.0 * config.sphere.radius;
    const Types::complex_d wave_number{Physics::get_k_on_frquency(config.frequency_ghz), 0.0};

    for (const Types::index Nx : config.mesh.points_per_axis) {
        const Types::index Ny = Nx;
        const Types::index Nz = Nx;
        Mesh::VolumeMesh::CubeMeshWithData mesh{
            Types::point_t{-cube_length / 2.0, -cube_length / 2.0, -cube_length / 2.0},
            cube_length,
            cube_length,
            cube_length,
            Nx,
            Ny,
            Nz};

        mesh.setName("sphere_" + std::to_string(Nx-1));
        std::cout << Nx - 1 << " cubes per size" << std::endl;

        mesh.smoothScalarData<DecartIntegration::NewtonCotess::Quadrature<1, 1, 1>>("eps", homogeneous_sphere);

        const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();

        std::cout.precision(16);
        std::cout << "Real sphere radius is\t\t\t\t " << config.sphere.radius << std::endl;

        Physics::planeWaveCase incident_field{config.incident_wave.polarization, wave_number,
                                              config.incident_wave.direction};
        std::cout << "Длина волны в свободном пространстве = " << 2.0 * M_PI / wave_number.real() << std::endl;
        std::cout << "lambda_0 / mesh.h = "
                  << 2.0 * M_PI / (wave_number.real() * cube_length / static_cast<Types::scalar>(Nx - 1)) << std::endl;
        std::cout << "lambda / mesh.h = "
                  << 2.0 * M_PI /
                         (std::sqrt(std::abs(config.sphere.epsilon)) * wave_number.real() * cube_length /
                          static_cast<Types::scalar>(Nx - 1))
                  << std::endl;

        const Types::scalar basis_function_module = 1.0 / std::sqrt(cube_measure);
        Operators::Volume::ProjectorOnMesh projector{mesh};
        const auto projected_rhs =
            projector([incident_field](Types::point_t point) { return incident_field.value(point); });
        Types::VectorXc rhs = projected_rhs * basis_function_module;

        Operators::Volume::operator_K_over_cube_mesh operator_k{wave_number, mesh};
        operator_k.set_tolerances(config.integration.relative_tolerance, config.integration.absolute_tolerance);
        operator_k.set_adaptive_integration_max_levels({config.integration.level_2d, config.integration.level_3d,
                                                        config.integration.level_4d, config.integration.level_6d});
        operator_k.set_nearness_threshold(config.integration.nearness_threshold);
        const auto operator_k_matrix = operator_k.compute_galerkin_matrix(basis_function_module);
        std::cout << "Matrix sizes: " << operator_k_matrix.cols() << " x " << operator_k_matrix.rows() << std::endl;

        using FourierOperator = Math::Fourier::TripleToeplitz3x3FourierParallel<Types::complex_d>;
        const FourierOperator fourier_operator{operator_k_matrix};

        Types::VectorXc epsilon_minus_one = Types::VectorXc::Zero(3 * mesh.getCells().size());
        const auto epsilon_data = mesh.getScalarData("eps");
        for (std::size_t cell = 0; cell < mesh.getCells().size(); ++cell) {
            const Types::complex_d contrast = epsilon_data[cell] - Types::complex_d{1.0, 0.0};
            epsilon_minus_one.segment<3>(3 * cell).setConstant(contrast);
        }

        auto analytical_solution = Research::VolDie::Mie::calculate_field_on_mesh(mesh, config.sphere.radius + additional_mie_radius,
                                                                                  config.sphere.epsilon, wave_number);
        mesh.setVectorData("analytical_solution", std::move(analytical_solution));

        Math::LinAgl::Matrix::Wrappers::VolumeOperatorMatrixReplacement system{fourier_operator, epsilon_minus_one};
        rhs = system.modify_rhs_according_to_mask(rhs);
        const Types::VectorXc solution = Research::solve<Eigen::GMRES>(system, rhs, config.gmres.max_iterations,
                                                                       config.gmres.tolerance, config.gmres.restart);

        std::vector<Types::Vector3c> field_on_mesh;
        field_on_mesh.reserve(mesh.getCells().size());
        for (std::size_t cell = 0; cell < mesh.getCells().size(); ++cell) {
            field_on_mesh.emplace_back(solution[3 * cell] * basis_function_module,
                                       solution[3 * cell + 1] * basis_function_module,
                                       solution[3 * cell + 2] * basis_function_module);
        }
        mesh.setVectorData("solution", std::move(field_on_mesh));

        // The snapshot contains both "solution" and "analytical_solution" vector fields.
        VTK::volume_mesh_withdata_snapshot(mesh, (config.output_directory / "").string());

        const int angle_count = config.angular_samples;
        const auto get_tau_hh = [](Types::scalar phi) { return Types::Vector3d{std::sin(phi), 0.0, std::cos(phi)}; };
        const auto get_tau_vv = [](Types::scalar phi) { return Types::Vector3d{0.0, std::sin(phi), std::cos(phi)}; };

        auto angle_view = std::views::iota(0, angle_count) |
                          std::views::transform([angle_count](int index) { return index * M_PI / angle_count; });
        std::vector<Types::scalar> phis{angle_view.begin(), angle_view.end()};
        std::vector<Types::scalar> rsp_hh(angle_count);
        std::vector<Types::scalar> rsp_vv(angle_count);
        std::vector<Types::scalar> analytical_rsp_hh(angle_count);
        std::vector<Types::scalar> analytical_rsp_vv(angle_count);

#pragma omp parallel for
        for (auto &&[vertical, horizontal, analytical_vertical, analytical_horizontal, phi] :
             std::views::zip(rsp_vv, rsp_hh, analytical_rsp_vv, analytical_rsp_hh, phis)) {
            vertical = 10.0 * std::log10(ESA::calculateRSP_kahan(get_tau_vv(phi), wave_number, "solution", mesh));
            horizontal = 10.0 * std::log10(ESA::calculateRSP_kahan(get_tau_hh(phi), wave_number, "solution", mesh));
            analytical_vertical =
                10.0 * std::log10(ESA::calculateRSP_kahan(get_tau_vv(phi), wave_number, "analytical_solution", mesh));
            analytical_horizontal =
                10.0 * std::log10(ESA::calculateRSP_kahan(get_tau_hh(phi), wave_number, "analytical_solution", mesh));
        }

        auto degree_view = angle_view | std::views::transform([](Types::scalar phi) { return phi * 180.0 / M_PI; });
        std::vector<Types::scalar> angles_degrees{degree_view.begin(), degree_view.end()};

        const auto rsp_vv_path = config.output_directory / ("sphere_sigma_vv_" + std::to_string(Nx-1) + ".csv");
        const auto rsp_hh_path = config.output_directory / ("sphere_sigma_hh_" + std::to_string(Nx-1) + ".csv");
        const auto analytical_rsp_vv_path =
            config.output_directory / ("an_sphere_sigma_vv_" + std::to_string(Nx-1) + ".csv");
        const auto analytical_rsp_hh_path =
            config.output_directory / ("an_sphere_sigma_hh_" + std::to_string(Nx-1) + ".csv");

        std::ofstream rsp_vv_file{rsp_vv_path};
        std::ofstream rsp_hh_file{rsp_hh_path};
        std::ofstream analytical_rsp_vv_file{analytical_rsp_vv_path};
        std::ofstream analytical_rsp_hh_file{analytical_rsp_hh_path};
        ensure_output_stream_is_open(rsp_vv_file, rsp_vv_path);
        ensure_output_stream_is_open(rsp_hh_file, rsp_hh_path);
        ensure_output_stream_is_open(analytical_rsp_vv_file, analytical_rsp_vv_path);
        ensure_output_stream_is_open(analytical_rsp_hh_file, analytical_rsp_hh_path);
        Utils::to_csv(angles_degrees, rsp_vv, "angle", "rsp", rsp_vv_file);
        Utils::to_csv(angles_degrees, rsp_hh, "angle", "rsp", rsp_hh_file);
        Utils::to_csv(angles_degrees, analytical_rsp_vv, "angle", "rsp", analytical_rsp_vv_file);
        Utils::to_csv(angles_degrees, analytical_rsp_hh, "angle", "rsp", analytical_rsp_hh_file);
    }

    const int angle_count = config.angular_samples;
    auto angle_view = std::views::iota(0, angle_count) |
                      std::views::transform([angle_count](int index) { return index * M_PI / angle_count; });
    std::vector<Types::scalar> phis{angle_view.begin(), angle_view.end()};
    auto degree_view = angle_view | std::views::transform([](Types::scalar phi) { return phi * 180.0 / M_PI; });
    std::vector<Types::scalar> angles_degrees{degree_view.begin(), degree_view.end()};

    auto mie_rsp = Research::VolDie::Mie::calculate_rsp(phis, config.sphere.radius + additional_mie_radius, config.sphere.epsilon, wave_number);
    for (Types::scalar &value : mie_rsp.vv) {
        value = 10.0 * std::log10(value) + config.analytical_rsp_db_offset;
    }
    for (Types::scalar &value : mie_rsp.hh) {
        value = 10.0 * std::log10(value) + config.analytical_rsp_db_offset;
    }

    const auto mie_rsp_vv_path = config.output_directory / "mie_sigma_vv.csv";
    const auto mie_rsp_hh_path = config.output_directory / "mie_sigma_hh.csv";
    std::ofstream mie_rsp_vv_file{mie_rsp_vv_path};
    std::ofstream mie_rsp_hh_file{mie_rsp_hh_path};
    ensure_output_stream_is_open(mie_rsp_vv_file, mie_rsp_vv_path);
    ensure_output_stream_is_open(mie_rsp_hh_file, mie_rsp_hh_path);
    Utils::to_csv(angles_degrees, mie_rsp.vv, "angle", "rsp", mie_rsp_vv_file);
    Utils::to_csv(angles_degrees, mie_rsp.hh, "angle", "rsp", mie_rsp_hh_file);

    return 0;
}
