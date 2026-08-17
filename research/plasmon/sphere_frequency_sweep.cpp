// Frequency sweep of extinction and radar cross sections of a homogeneous sphere.

#include "EMW/types/Types.hpp"

#include "experiment/ESA.hpp"
#include "experiment/PhysicalCondition.hpp"
#include "math/fourier/TripleToeplitz3x3Fourier.hpp"
#include "math/integration/decart/Integration.hpp"
#include "mesh/volume_mesh/CubeMeshWithData.hpp"
#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "../Solve.hpp"
#include "../cluster_dielectric/MatrixReplacement.hpp"
#include "../cluster_dielectric/MatrixTraits.hpp"
#include "configs/SphereFrequencySweepConfig.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace EMW;

namespace {

using Configuration = Research::Plasmon::SphereFrequencySweepConfig;

struct FrequencyResult {
    Types::scalar wave_number{};
    Types::index cells_per_axis{};
    Types::scalar cell_size{};
    Types::scalar cells_per_internal_wavelength{};
    Types::scalar extinction_cross_section{};
    Types::scalar forward_rcs{};
    Types::scalar backward_rcs{};
    std::filesystem::path bistatic_file;
};

void ensure_output_stream_is_open(const std::ofstream &stream, const std::filesystem::path &path) {
    if (!stream) {
        throw std::runtime_error("Cannot open output file '" + path.string() + "'");
    }
}

Types::index frequency_count(const Configuration &config) {
    const Types::scalar count =
        std::round((config.frequency_sweep.end_ghz - config.frequency_sweep.start_ghz) /
                   config.frequency_sweep.step_ghz) +
        1.0;
    if (!std::isfinite(count) || count > static_cast<Types::scalar>(std::numeric_limits<Types::index>::max())) {
        throw std::overflow_error("The frequency sweep contains too many points");
    }
    return static_cast<Types::index>(count);
}

Types::index cells_per_axis(Types::scalar wave_number, const Configuration &config) {
    // Enforce dx <= lambda_inside/N, where
    // lambda_inside = 2*pi/(k_0*sqrt(|epsilon_r|)).
    const Types::scalar cube_length = 2.0 * config.sphere.radius;
    const Types::scalar internal_wave_number = wave_number * std::sqrt(std::abs(config.sphere.epsilon));
    const Types::scalar required_cells =
        std::ceil(config.mesh.cells_per_internal_wavelength * cube_length * internal_wave_number /
                  (2.0 * std::numbers::pi_v<Types::scalar>));
    if (!std::isfinite(required_cells) ||
        required_cells > static_cast<Types::scalar>(std::numeric_limits<Types::index>::max() - 1)) {
        throw std::overflow_error("The required mesh size is out of range");
    }
    return std::max(config.mesh.min_cells_per_axis, static_cast<Types::index>(required_cells));
}

Types::scalar to_dbsm(Types::scalar cross_section) {
    constexpr Types::scalar floor = std::numeric_limits<Types::scalar>::min();
    return 10.0 * std::log10(std::max(cross_section, floor));
}

std::string frequency_file_tag(Types::index frequency_index, Types::scalar frequency_ghz) {
    std::ostringstream tag;
    tag << std::setw(6) << std::setfill('0') << frequency_index << '_' << std::fixed << std::setprecision(9)
        << frequency_ghz;
    std::string result = tag.str();
    std::replace(result.begin(), result.end(), '.', 'p');
    return result;
}

Types::scalar calculate_extinction_cross_section(const Types::VectorXc &incident_projection,
                                                 const Types::VectorXc &solution,
                                                 const Types::VectorXc &epsilon_minus_one,
                                                 Types::scalar wave_number,
                                                 const Types::Vector3d &polarization) {
    // The basis is L2-normalized, so this scalar product approximates
    // integral_V conj(E_inc) . (epsilon_r - 1) E dV.
    const Types::complex_d extinction_integral =
        incident_projection.dot(epsilon_minus_one.cwiseProduct(solution));
    return wave_number * std::imag(extinction_integral) / polarization.squaredNorm();
}

std::vector<Types::Vector3c> coefficients_to_cell_field(const Types::VectorXc &coefficients,
                                                        Types::scalar basis_function_module,
                                                        Types::index cell_count) {
    std::vector<Types::Vector3c> field;
    field.reserve(cell_count);
    for (Types::index cell = 0; cell < cell_count; ++cell) {
        field.emplace_back(coefficients[3 * cell] * basis_function_module,
                           coefficients[3 * cell + 1] * basis_function_module,
                           coefficients[3 * cell + 2] * basis_function_module);
    }
    return field;
}

std::filesystem::path write_bistatic_diagram(const Mesh::VolumeMesh::CubeMeshWithData &mesh,
                                             Types::complex_d wave_number,
                                             Types::index frequency_index,
                                             Types::scalar frequency_ghz,
                                             const Configuration &config) {
    const Types::index polar_samples = config.bistatic_diagram.polar_samples;
    const Types::index azimuthal_samples = config.bistatic_diagram.azimuthal_samples;
    if (polar_samples > std::numeric_limits<Types::index>::max() / azimuthal_samples) {
        throw std::overflow_error("The bistatic angular grid is too large");
    }
    const Types::index sample_count = polar_samples * azimuthal_samples;
    std::vector<Types::scalar> rcs(sample_count);

#pragma omp parallel for schedule(dynamic)
    for (Types::integer linear_index = 0; linear_index < static_cast<Types::integer>(sample_count);
         ++linear_index) {
        const Types::index sample = static_cast<Types::index>(linear_index);
        const Types::index polar_index = sample / azimuthal_samples;
        const Types::index azimuthal_index = sample % azimuthal_samples;
        const Types::scalar theta = std::numbers::pi_v<Types::scalar> * polar_index /
                                    static_cast<Types::scalar>(polar_samples - 1);
        const Types::scalar phi = 2.0 * std::numbers::pi_v<Types::scalar> * azimuthal_index /
                                  static_cast<Types::scalar>(azimuthal_samples);
        const Types::Vector3d direction{std::sin(theta) * std::cos(phi),
                                        std::sin(theta) * std::sin(phi),
                                        std::cos(theta)};
        rcs[sample] = ESA::calculateRSP_kahan(direction, wave_number, "solution", mesh);
    }

    const std::filesystem::path path =
        config.output.directory /
        (config.output.bistatic_file_prefix + "_" + frequency_file_tag(frequency_index, frequency_ghz) +
         "_GHz.csv");
    std::ofstream output{path};
    ensure_output_stream_is_open(output, path);
    output << "theta_deg,phi_deg,direction_x,direction_y,direction_z,rcs_m2,rcs_dbsm\n";
    output << std::setprecision(16);
    for (Types::index polar_index = 0; polar_index < polar_samples; ++polar_index) {
        const Types::scalar theta = std::numbers::pi_v<Types::scalar> * polar_index /
                                    static_cast<Types::scalar>(polar_samples - 1);
        for (Types::index azimuthal_index = 0; azimuthal_index < azimuthal_samples; ++azimuthal_index) {
            const Types::scalar phi = 2.0 * std::numbers::pi_v<Types::scalar> * azimuthal_index /
                                      static_cast<Types::scalar>(azimuthal_samples);
            const Types::Vector3d direction{std::sin(theta) * std::cos(phi),
                                            std::sin(theta) * std::sin(phi),
                                            std::cos(theta)};
            const Types::scalar value = rcs[polar_index * azimuthal_samples + azimuthal_index];
            output << theta * 180.0 / std::numbers::pi_v<Types::scalar> << ','
                   << phi * 180.0 / std::numbers::pi_v<Types::scalar> << ',' << direction.x() << ','
                   << direction.y() << ',' << direction.z() << ',' << value << ',' << to_dbsm(value) << '\n';
        }
    }
    return path;
}

FrequencyResult calculate_at_frequency(Types::index frequency_index,
                                       Types::scalar frequency_ghz,
                                       const Configuration &config) {
    const auto homogeneous_sphere = [&config](const Types::point_t &point) {
        return point.norm() < config.sphere.radius ? config.sphere.epsilon
                                                   : Types::complex_d{1.0, 0.0};
    };

    const Types::complex_d wave_number{Physics::get_k_on_frquency(frequency_ghz), 0.0};
    const Types::scalar cube_length = 2.0 * config.sphere.radius;
    const Types::index cell_count = cells_per_axis(wave_number.real(), config);
    const Types::index points_per_axis = cell_count + 1;
    Mesh::VolumeMesh::CubeMeshWithData mesh{
        Types::point_t{-cube_length / 2.0, -cube_length / 2.0, -cube_length / 2.0},
        cube_length,
        cube_length,
        cube_length,
        points_per_axis,
        points_per_axis,
        points_per_axis};
    mesh.setName("sphere_" + frequency_file_tag(frequency_index, frequency_ghz) + "_GHz");
    mesh.smoothScalarData<DecartIntegration::GaussLegendre::Quadrature<1, 1, 1>>("eps", homogeneous_sphere);

    const Types::scalar internal_wavelength =
        2.0 * std::numbers::pi_v<Types::scalar> /
        (wave_number.real() * std::sqrt(std::abs(config.sphere.epsilon)));
    const Types::scalar actual_cells_per_wavelength = internal_wavelength / mesh.dx();
    const Types::scalar mesh_tolerance =
        64.0 * std::numeric_limits<Types::scalar>::epsilon() * actual_cells_per_wavelength;
    if (actual_cells_per_wavelength + mesh_tolerance < config.mesh.cells_per_internal_wavelength) {
        throw std::runtime_error("The generated mesh violates cells_per_internal_wavelength");
    }

    std::cout << "\nFrequency = " << frequency_ghz << " GHz\n"
              << "cells per axis = " << cell_count << ", points per axis = " << points_per_axis << '\n'
              << "cell size = " << mesh.dx() << " m\n"
              << "cells per internal wavelength = " << actual_cells_per_wavelength << std::endl;

    Physics::planeWaveCase incident_field{
        config.incident_wave.polarization, wave_number, config.incident_wave.direction};
    const Types::scalar cell_measure = mesh.dx() * mesh.dy() * mesh.dz();
    const Types::scalar basis_function_module = 1.0 / std::sqrt(cell_measure);
    Operators::Volume::ProjectorOnMesh projector{mesh};
    const auto projected_rhs =
        projector([incident_field](Types::point_t point) { return incident_field.value(point); });
    const Types::VectorXc incident_projection = projected_rhs * basis_function_module;

    Operators::Volume::operator_K_over_cube_mesh operator_k{wave_number, mesh};
    operator_k.set_tolerances(config.integration.relative_tolerance,
                              config.integration.absolute_tolerance);
    operator_k.set_adaptive_integration_max_levels({config.integration.level_2d,
                                                    config.integration.level_3d,
                                                    config.integration.level_4d,
                                                    config.integration.level_6d});
    operator_k.set_nearness_threshold(config.integration.nearness_threshold);
    const auto operator_k_matrix = operator_k.compute_galerkin_matrix(basis_function_module);
    std::cout << "Matrix sizes: " << operator_k_matrix.rows() << " x " << operator_k_matrix.cols()
              << std::endl;

    using FourierOperator = Math::Fourier::TripleToeplitz3x3FourierParallel<Types::complex_d>;
    const FourierOperator fourier_operator{operator_k_matrix};
    Types::VectorXc epsilon_minus_one = Types::VectorXc::Zero(3 * mesh.getCells().size());
    const auto epsilon_data = mesh.getScalarData("eps");
    for (Types::index cell = 0; cell < mesh.getCells().size(); ++cell) {
        const Types::complex_d contrast = epsilon_data[cell] - Types::complex_d{1.0, 0.0};
        epsilon_minus_one.segment<3>(3 * cell).setConstant(contrast);
    }

    Math::LinAgl::Matrix::Wrappers::VolumeOperatorMatrixReplacement system{fourier_operator,
                                                                            epsilon_minus_one};
    const Types::VectorXc rhs = system.modify_rhs_according_to_mask(incident_projection);
    const Types::VectorXc solution = Research::solve<Eigen::GMRES>(system,
                                                                   rhs,
                                                                   config.gmres.max_iterations,
                                                                   config.gmres.tolerance,
                                                                   config.gmres.restart);

    const Types::scalar extinction_cross_section = calculate_extinction_cross_section(
        incident_projection,
        solution,
        system.get_epsilon_vec(),
        wave_number.real(),
        config.incident_wave.polarization);
    mesh.setVectorData(
        "solution", coefficients_to_cell_field(solution, basis_function_module, mesh.getCells().size()));

    // The configured plane-wave direction is treated as the forward observation direction.
    const Types::scalar forward_rcs =
        ESA::calculateRSP_kahan(config.incident_wave.direction, wave_number, "solution", mesh);
    const Types::scalar backward_rcs =
        ESA::calculateRSP_kahan(-config.incident_wave.direction, wave_number, "solution", mesh);
    const std::filesystem::path bistatic_file =
        write_bistatic_diagram(mesh, wave_number, frequency_index, frequency_ghz, config);

    std::cout << "extinction cross section = " << extinction_cross_section << " m^2\n"
              << "forward RCS = " << forward_rcs << " m^2 (" << to_dbsm(forward_rcs) << " dBsm)\n"
              << "backward RCS = " << backward_rcs << " m^2 (" << to_dbsm(backward_rcs) << " dBsm)\n"
              << "bistatic diagram written to " << bistatic_file << std::endl;

    return {.wave_number = wave_number.real(),
            .cells_per_axis = cell_count,
            .cell_size = mesh.dx(),
            .cells_per_internal_wavelength = actual_cells_per_wavelength,
            .extinction_cross_section = extinction_cross_section,
            .forward_rcs = forward_rcs,
            .backward_rcs = backward_rcs,
            .bistatic_file = bistatic_file};
}

} // namespace

int main(int argc, char **argv) {
    Eigen::setNbThreads(1);

    try {
        const Configuration config =
            Research::Plasmon::load_sphere_frequency_sweep_config(argc, argv);
        std::filesystem::create_directories(config.output.directory);
        const std::filesystem::path summary_path = config.output.directory / config.output.summary_file;
        std::ofstream summary{summary_path};
        ensure_output_stream_is_open(summary, summary_path);
        summary << "frequency_GHz,wave_number_1_m,cells_per_axis,points_per_axis,cell_size_m,"
                   "cells_per_internal_wavelength,extinction_cross_section_m2,forward_rcs_m2,"
                   "forward_rcs_dbsm,backward_rcs_m2,backward_rcs_dbsm,bistatic_file\n";
        summary << std::setprecision(16);

        const Types::index count = frequency_count(config);
        for (Types::index frequency_index = 0; frequency_index < count; ++frequency_index) {
            const Types::scalar frequency_ghz =
                config.frequency_sweep.start_ghz + frequency_index * config.frequency_sweep.step_ghz;
            const FrequencyResult result =
                calculate_at_frequency(frequency_index, frequency_ghz, config);
            summary << frequency_ghz << ',' << result.wave_number << ',' << result.cells_per_axis << ','
                    << result.cells_per_axis + 1 << ',' << result.cell_size << ','
                    << result.cells_per_internal_wavelength << ',' << result.extinction_cross_section << ','
                    << result.forward_rcs << ',' << to_dbsm(result.forward_rcs) << ',' << result.backward_rcs
                    << ',' << to_dbsm(result.backward_rcs) << ',' << result.bistatic_file.filename().string()
                    << '\n';
            summary.flush();
        }

        std::cout << "\nFrequency sweep summary written to " << summary_path << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "sphere_frequency_sweep: " << error.what() << std::endl;
        return 1;
    }
}
