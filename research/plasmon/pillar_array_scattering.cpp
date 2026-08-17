// Scattering of a plane wave by dielectric pillars on a finite dielectric substrate.

#include "PillarArrayGeometry.hpp"
#include "configs/PillarArrayScatteringConfig.hpp"

#include "EMW/Utils.hpp"
#include "EMW/types/Types.hpp"

#include "experiment/ESA.hpp"
#include "experiment/PhysicalCondition.hpp"
#include "mesh/volume_mesh/CubeMeshWithData.hpp"
#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"
#include "visualisation/include/VTKFunctions.hpp"

#include "../Solve.hpp"
#include "../cluster_dielectric/MatrixReplacement.hpp"
#include "../cluster_dielectric/MatrixTraits.hpp"

#include "math/fourier/TripleToeplitz3x3Fourier.hpp"
#include "math/MathConstants.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace EMW;

namespace {

std::vector<Types::Vector3c> coefficients_to_cell_field(const Types::VectorXc &coefficients,
                                                        Types::scalar basis_function_module) {
    std::vector<Types::Vector3c> field;
    field.reserve(static_cast<std::size_t>(coefficients.size() / 3));
    for (Types::index cell = 0; cell < coefficients.size() / 3; ++cell) {
        field.emplace_back(coefficients[3 * cell] * basis_function_module,
                           coefficients[3 * cell + 1] * basis_function_module,
                           coefficients[3 * cell + 2] * basis_function_module);
    }
    return field;
}

void ensure_output_stream_is_open(const std::ofstream &stream, const std::filesystem::path &path) {
    if (!stream) {
        throw std::runtime_error("Cannot open output file '" + path.string() + "'");
    }
}

Types::scalar to_decibels(Types::scalar value) {
    return 10.0 * std::log10(std::max(value, std::numeric_limits<Types::scalar>::min()));
}

} // namespace

int main(int argc, char **argv) {
    Eigen::setNbThreads(1);

    const auto config = Research::Plasmon::load_pillar_array_scattering_config(argc, argv);
    const auto mesh_description = Research::Plasmon::make_pillar_array_mesh_description(config);
    constexpr Types::scalar model_wavelength = 1.0;
    const Types::scalar model_meters_per_nanometer = model_wavelength / config.wavelength_nm;
    const Types::scalar physical_meters_per_model_meter = config.wavelength_nm * 1.0e-9 / model_wavelength;
    const Types::scalar physical_frequency_hz = Math::Constants::c / (config.wavelength_nm * 1.0e-9);
    const Types::scalar model_frequency_hz = Math::Constants::c / model_wavelength;
    const Types::complex_d wave_number{2.0 * M_PI / model_wavelength, 0.0};

    Mesh::VolumeMesh::CubeMeshWithData mesh{
        mesh_description.min_corner, mesh_description.size_x, mesh_description.size_y, mesh_description.size_z,
        mesh_description.cells_x + 1, mesh_description.cells_y + 1, mesh_description.cells_z + 1};
    mesh.setName("pillar_array_scattering");

    auto permittivity = Research::Plasmon::make_pillar_array_permittivity(mesh, config);
    const Types::VectorXc epsilon_minus_one =
        Research::Plasmon::make_pillar_array_epsilon_minus_one(permittivity);

    const Types::scalar mesh_step_nm = config.geometry.lattice.cell_size_nm.x() /
                                       static_cast<Types::scalar>(config.mesh.cells_per_geometry_cell_side);
    const Types::index substrate_layers = static_cast<Types::index>(
        std::llround(config.geometry.substrate_thickness_nm / mesh_step_nm));
    const Types::index pillar_cells_x =
        static_cast<Types::index>(std::llround(config.geometry.pillar_size_nm.x() / mesh_step_nm));
    const Types::index pillar_cells_y =
        static_cast<Types::index>(std::llround(config.geometry.pillar_size_nm.y() / mesh_step_nm));
    const Types::index pillar_cells_z =
        static_cast<Types::index>(std::llround(config.geometry.pillar_size_nm.z() / mesh_step_nm));
    const Types::index substrate_cells = mesh_description.cells_x * mesh_description.cells_y * substrate_layers;
    const Types::index pillar_cells =
        config.geometry.pillar_centers.size() * pillar_cells_x * pillar_cells_y * pillar_cells_z;
    const Types::index air_cells = mesh.getCells().size() - substrate_cells - pillar_cells;
    mesh.setScalarData("eps", std::move(permittivity));

    std::cout << "Physical free-space wavelength = " << config.wavelength_nm << " nm\n"
              << "Physical frequency = " << physical_frequency_hz << " Hz\n"
              << "Equivalent-model wavelength = " << model_wavelength << " m\n"
              << "Scale = " << model_meters_per_nanometer << " model m/nm\n"
              << "Physical/model length scale = " << physical_meters_per_model_meter << '\n'
              << "Equivalent-model frequency = " << model_frequency_hz << " Hz\n"
              << "Equivalent-model wave number = " << wave_number.real() << " 1/m\n"
              << "Geometry lattice = " << config.geometry.lattice.cells[0] << " x "
              << config.geometry.lattice.cells[1] << " cells\n"
              << "Pillars = " << config.geometry.pillar_centers.size() << '\n'
              << "Volume mesh = " << mesh_description.cells_x << " x " << mesh_description.cells_y << " x "
              << mesh_description.cells_z << " cubic cells\n"
              << "Volume-mesh step = " << mesh_description.step << " m\n"
              << "Material cell counts: substrate = " << substrate_cells << ", pillars = " << pillar_cells
              << ", air = " << air_cells << std::endl;

    const Physics::planeWaveCase incident_wave{config.incident_wave.polarization, wave_number,
                                               config.incident_wave.direction};
    const Types::scalar cell_measure = mesh.dx() * mesh.dy() * mesh.dz();
    const Types::scalar basis_function_module = 1.0 / std::sqrt(cell_measure);

    Operators::Volume::ProjectorOnMesh projector{mesh};
    const Types::VectorXc incident_coefficients =
        projector([incident_wave](const Types::point_t &point) { return incident_wave.value(point); }) *
        basis_function_module;
    Types::VectorXc rhs = incident_coefficients;

    Operators::Volume::operator_K_over_cube_mesh operator_k{wave_number, mesh};
    operator_k.set_tolerances(config.integration.relative_tolerance, config.integration.absolute_tolerance);
    operator_k.set_adaptive_integration_max_levels({config.integration.level_2d, config.integration.level_3d,
                                                    config.integration.level_4d, config.integration.level_6d});
    operator_k.set_nearness_threshold(config.integration.nearness_threshold);
    const auto operator_k_matrix = operator_k.compute_galerkin_matrix(basis_function_module);
    std::cout << "Matrix sizes: " << operator_k_matrix.rows() << " x " << operator_k_matrix.cols() << std::endl;

    using FourierOperator = Math::Fourier::TripleToeplitz3x3FourierParallel<Types::complex_d>;
    const FourierOperator fourier_operator{operator_k_matrix};
    Math::LinAgl::Matrix::Wrappers::VolumeOperatorMatrixReplacement system{fourier_operator, epsilon_minus_one};
    rhs = system.modify_rhs_according_to_mask(rhs);
    const Types::VectorXc solution = Research::solve<Eigen::GMRES>(system, rhs, config.gmres.max_iterations,
                                                                   config.gmres.tolerance, config.gmres.restart);

    mesh.setVectorData("incident_field", coefficients_to_cell_field(incident_coefficients, basis_function_module));
    mesh.setVectorData("solution", coefficients_to_cell_field(solution, basis_function_module));

    std::filesystem::create_directories(config.output_directory);
    VTK::volume_mesh_withdata_snapshot(mesh, (config.output_directory / "").string());

    const Types::Vector3d substrate_normal{0.0, 0.0, 1.0};
    const Types::Vector3d incidence_plane_normal =
        config.incident_wave.direction.cross(substrate_normal).normalized();
    const Types::Vector3d positive_angle_direction =
        incidence_plane_normal.cross(config.incident_wave.direction).normalized();

    std::vector<Types::scalar> angles_deg(config.bistatic_rcs.samples);
    std::vector<Types::scalar> bistatic_rcs_db(config.bistatic_rcs.samples);
    const Types::scalar angle_step =
        (config.bistatic_rcs.phi_max_deg - config.bistatic_rcs.phi_min_deg) /
        static_cast<Types::scalar>(config.bistatic_rcs.samples - 1);

#pragma omp parallel for
    for (Types::index sample = 0; sample < config.bistatic_rcs.samples; ++sample) {
        const Types::scalar angle_deg = config.bistatic_rcs.phi_min_deg + sample * angle_step;
        const Types::scalar angle_rad = angle_deg * M_PI / 180.0;
        const Types::Vector3d observation_direction =
            std::cos(angle_rad) * config.incident_wave.direction +
            std::sin(angle_rad) * positive_angle_direction;
        angles_deg[sample] = angle_deg;
        const Types::scalar model_rcs =
            ESA::calculateRSP_kahan(observation_direction, wave_number, "solution", mesh);
        const Types::scalar physical_rcs =
            model_rcs * physical_meters_per_model_meter * physical_meters_per_model_meter;
        bistatic_rcs_db[sample] = to_decibels(physical_rcs);
    }

    const std::filesystem::path rcs_path = config.output_directory / "pillar_array_bistatic_rcs.csv";
    std::ofstream rcs_file{rcs_path};
    ensure_output_stream_is_open(rcs_file, rcs_path);
    Utils::to_csv(angles_deg, bistatic_rcs_db, "phi_from_incident_wave_deg", "bistatic_rcs_dbsm", rcs_file);
    std::cout << "Bistatic RCS written to " << rcs_path << std::endl;

    return 0;
}
