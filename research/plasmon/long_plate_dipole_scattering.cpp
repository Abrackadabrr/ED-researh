// Scattering of a line-source field by a long dielectric plate.

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

#include "math/fourier/TripleToeplitz3x3Fourier.hpp"
#include "math/integration/decart/Integration.hpp"

#include "EMW/Utils.hpp"
#include "configs/LongPlateDipoleScatteringConfig.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace EMW;

namespace {

Types::index cells_for_dimension(Types::scalar dimension, Types::scalar internal_wavelength,
                                 Types::scalar cells_per_internal_wavelength) {
    const Types::scalar requested_cells = dimension * cells_per_internal_wavelength / internal_wavelength;
    return std::max<Types::index>(1, static_cast<Types::index>(std::ceil(requested_cells - 1e-12)));
}

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

bool coordinate_is_inside(Types::scalar value, Types::scalar min_value, Types::scalar max_value) {
    const Types::scalar tolerance = 64.0 * std::numeric_limits<Types::scalar>::epsilon() *
                                    std::max({1.0, std::abs(min_value), std::abs(max_value)});
    return value >= min_value - tolerance && value <= max_value + tolerance;
}

bool coordinate_is_on_mesh_plane(Types::scalar value, Types::scalar min_value, Types::scalar cell_size,
                                 Types::index cell_count) {
    const Types::scalar relative_coordinate = (value - min_value) / cell_size;
    const Types::scalar tolerance =
        64.0 * std::numeric_limits<Types::scalar>::epsilon() * std::max(1.0, std::abs(relative_coordinate));
    return relative_coordinate >= -tolerance && relative_coordinate <= cell_count + tolerance &&
           std::abs(relative_coordinate - std::round(relative_coordinate)) <= tolerance;
}

bool point_is_on_cell_face(const Types::point_t &point,
                           const Research::Plasmon::LongPlateDipoleScatteringConfig &config,
                           const Mesh::VolumeMesh::CubeMesh &mesh, Types::index cells_x, Types::index cells_y,
                           Types::index cells_z) {
    const Types::point_t plate_max =
        config.plate.min_corner + Types::point_t{config.plate.length, config.plate.width, config.plate.thickness};
    const bool inside_x = coordinate_is_inside(point.x(), config.plate.min_corner.x(), plate_max.x());
    const bool inside_y = coordinate_is_inside(point.y(), config.plate.min_corner.y(), plate_max.y());
    const bool inside_z = coordinate_is_inside(point.z(), config.plate.min_corner.z(), plate_max.z());

    return (inside_y && inside_z &&
            coordinate_is_on_mesh_plane(point.x(), config.plate.min_corner.x(), mesh.dx(), cells_x)) ||
           (inside_x && inside_z &&
            coordinate_is_on_mesh_plane(point.y(), config.plate.min_corner.y(), mesh.dy(), cells_y)) ||
           (inside_x && inside_y &&
            coordinate_is_on_mesh_plane(point.z(), config.plate.min_corner.z(), mesh.dz(), cells_z));
}

} // namespace

int main(int argc, char **argv) {
    Eigen::setNbThreads(1);

    const auto config = Research::Plasmon::load_long_plate_dipole_scattering_config(argc, argv);

    const Types::complex_d wave_number{2.0 * M_PI / config.wavelength, 0.0};
    const Types::scalar source_length = config.source.length_wavelengths * config.wavelength;
    const Types::Vector3d source_segment = config.source.direction.normalized() * source_length;
    // For complex media, use |sqrt(epsilon_r)| = sqrt(|epsilon_r|), consistently with the other JVIE experiments.
    const Types::scalar internal_refractive_index = std::sqrt(std::abs(config.plate.epsilon));
    const Types::scalar internal_wavelength = config.wavelength / internal_refractive_index;
    const Types::index cells_x =
        cells_for_dimension(config.plate.length, internal_wavelength, config.mesh.cells_per_internal_wavelength);
    const Types::index cells_y =
        cells_for_dimension(config.plate.width, internal_wavelength, config.mesh.cells_per_internal_wavelength);
    const Types::index cells_z =
        cells_for_dimension(config.plate.thickness, internal_wavelength, config.mesh.cells_per_internal_wavelength);
    const Types::index Nx = cells_x + 1;
    const Types::index Ny = cells_y + 1;
    const Types::index Nz = cells_z + 1;

    Mesh::VolumeMesh::CubeMeshWithData mesh{
        config.plate.min_corner, config.plate.length, config.plate.width, config.plate.thickness, Nx, Ny, Nz};
    mesh.setName("long_plate_dipole_scattering");

    const auto homogeneous_plate = [&config](const Types::point_t &) { return config.plate.epsilon; };
    mesh.smoothScalarData<DecartIntegration::NewtonCotess::Quadrature<1, 1, 1>>("eps", homogeneous_plate);

    std::cout << "Free-space wavelength = " << config.wavelength << " m\n"
              << "Internal wavelength = " << internal_wavelength << " m\n"
              << "Requested cells per internal wavelength = " << config.mesh.cells_per_internal_wavelength << '\n'
              << "Wave number = " << wave_number.real() << " 1/m\n"
              << "Plate cells = " << cells_x << " x " << cells_y << " x " << cells_z << '\n'
              << "Cell sizes = " << mesh.dx() << " x " << mesh.dy() << " x " << mesh.dz() << " m\n"
              << "Actual cells per internal wavelength = " << internal_wavelength / mesh.dx() << " x "
              << internal_wavelength / mesh.dy() << " x " << internal_wavelength / mesh.dz() << '\n'
              << "Line-source center = " << config.source.center.transpose() << '\n'
              << "Line-source direction = " << config.source.direction.normalized().transpose() << '\n'
              << "Line-source length = " << source_length << " m (" << config.source.length_wavelengths
              << " wavelengths)" << std::endl;

    const Physics::LineSource line_source{source_segment, config.source.center, config.source.current_amplitude,
                                          wave_number};
    const Types::scalar cell_measure = mesh.dx() * mesh.dy() * mesh.dz();
    const Types::scalar basis_function_module = 1.0 / std::sqrt(cell_measure);

    Operators::Volume::ProjectorOnMesh projector{mesh};
    const Types::VectorXc incident_coefficients =
        projector([line_source](const Types::point_t &point) { return line_source(point); }) * basis_function_module;
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

    Types::VectorXc epsilon_minus_one = Types::VectorXc::Zero(3 * mesh.getCells().size());
    const Types::complex_d contrast = config.plate.epsilon - Types::complex_d{1.0, 0.0};
    for (std::size_t cell = 0; cell < mesh.getCells().size(); ++cell) {
        epsilon_minus_one.segment<3>(3 * cell).setConstant(contrast);
    }

    Math::LinAgl::Matrix::Wrappers::VolumeOperatorMatrixReplacement system{fourier_operator, epsilon_minus_one};
    rhs = system.modify_rhs_according_to_mask(rhs);
    const Types::VectorXc solution = Research::solve<Eigen::GMRES>(system, rhs, config.gmres.max_iterations,
                                                                   config.gmres.tolerance, config.gmres.restart);
    const Types::VectorXc scattered_coefficients = solution - incident_coefficients;

    auto solution_field = coefficients_to_cell_field(solution, basis_function_module);
    std::vector<Types::complex_d> total_field_magnitude;
    total_field_magnitude.reserve(solution_field.size());
    for (const Types::Vector3c &field_value : solution_field) {
        total_field_magnitude.emplace_back(field_value.norm(), 0.0);
    }

    mesh.setVectorData("incident_field", coefficients_to_cell_field(incident_coefficients, basis_function_module));
    mesh.setVectorData("solution", std::move(solution_field));
    mesh.setVectorData("scattered_field", coefficients_to_cell_field(scattered_coefficients, basis_function_module));
    mesh.setScalarData("total_field_magnitude", std::move(total_field_magnitude));

    VTK::volume_mesh_withdata_snapshot(mesh, (config.output_directory / "").string());

    // Calculate the total field on the regular point grid configured in YAML.
    const Types::index field_points_x = config.field_plane.points_x;
    const Types::index field_points_z = config.field_plane.points_z;
    const Types::scalar field_step_x =
        (config.field_plane.x_max - config.field_plane.x_min) / static_cast<Types::scalar>(field_points_x - 1);
    const Types::scalar field_step_z =
        (config.field_plane.z_max - config.field_plane.z_min) / static_cast<Types::scalar>(field_points_z - 1);

    std::vector<Types::point_t> field_points;
    field_points.reserve(static_cast<std::size_t>(field_points_x * field_points_z));
    std::size_t skipped_field_points = 0;
    for (Types::index z_index = 0; z_index < field_points_z; ++z_index) {
        for (Types::index x_index = 0; x_index < field_points_x; ++x_index) {
            const Types::point_t point{config.field_plane.x_min + x_index * field_step_x, config.field_plane.y,
                                       config.field_plane.z_min + z_index * field_step_z};
            if (point_is_on_cell_face(point, config, mesh, cells_x, cells_y, cells_z)) {
                ++skipped_field_points;
                continue;
            }
            field_points.emplace_back(point);
        }
    }

    auto polarization_field = mesh.getVectorData("solution");
    for (Types::Vector3c &field_value : polarization_field) {
        field_value *= contrast;
    }

    std::vector<Types::Vector3c> incident_field_on_xz(field_points.size());
    std::vector<Types::Vector3c> scattered_field_on_xz(field_points.size());
    std::vector<Types::Vector3c> total_field_on_xz(field_points.size());
    std::vector<Types::scalar> total_field_magnitude_on_xz(field_points.size());

#pragma omp parallel for
    for (std::size_t point_index = 0; point_index < field_points.size(); ++point_index) {
        incident_field_on_xz[point_index] = line_source(field_points[point_index]);
        scattered_field_on_xz[point_index] =
            operator_k.compute_inner_point(field_points[point_index], polarization_field);
        total_field_on_xz[point_index] = incident_field_on_xz[point_index] + scattered_field_on_xz[point_index];
        total_field_magnitude_on_xz[point_index] = total_field_on_xz[point_index].norm();
    }

    std::cout << "Field grid in xz plane: " << field_points_x << " x " << field_points_z
              << ", y = " << config.field_plane.y << " m, steps = " << field_step_x << " x " << field_step_z << " m"
              << ", retained points = " << field_points.size()
              << ", skipped points on cell faces = " << skipped_field_points << std::endl;
    VTK::field_in_points_snapshot({incident_field_on_xz, scattered_field_on_xz, total_field_on_xz},
                                  {total_field_magnitude_on_xz}, {"incident_field", "scattered_field", "total_field"},
                                  {"total_field_magnitude"}, field_points, "long_plate_field_xz",
                                  (config.output_directory / "").string());

    const int angle_count = config.angular_samples;
    auto angle_view = std::views::iota(0, angle_count) | std::views::transform([angle_count](int index) {
                          return 2.0 * M_PI * static_cast<Types::scalar>(index) / angle_count;
                      });
    std::vector<Types::scalar> angles{angle_view.begin(), angle_view.end()};
    std::vector<Types::scalar> pattern_xy_db(angle_count);
    std::vector<Types::scalar> pattern_xz_db(angle_count);

#pragma omp parallel for
    for (int index = 0; index < angle_count; ++index) {
        const Types::scalar angle = angles[static_cast<std::size_t>(index)];
        const Types::Vector3d direction_xy{std::cos(angle), std::sin(angle), 0.0};
        const Types::Vector3d direction_xz{std::cos(angle), 0.0, std::sin(angle)};
        pattern_xy_db[static_cast<std::size_t>(index)] =
            to_decibels(ESA::calculateRSP_kahan(direction_xy, wave_number, "solution", mesh));
        pattern_xz_db[static_cast<std::size_t>(index)] =
            to_decibels(ESA::calculateRSP_kahan(direction_xz, wave_number, "solution", mesh));
    }

    auto degree_view = angles | std::views::transform([](Types::scalar angle) { return angle * 180.0 / M_PI; });
    std::vector<Types::scalar> angles_degrees{degree_view.begin(), degree_view.end()};

    const auto pattern_xy_path = config.output_directory / "long_plate_scattered_pattern_xy.csv";
    const auto pattern_xz_path = config.output_directory / "long_plate_scattered_pattern_xz.csv";
    std::ofstream pattern_xy_file{pattern_xy_path};
    std::ofstream pattern_xz_file{pattern_xz_path};
    ensure_output_stream_is_open(pattern_xy_file, pattern_xy_path);
    ensure_output_stream_is_open(pattern_xz_file, pattern_xz_path);
    Utils::to_csv(angles_degrees, pattern_xy_db, "angle_deg", "scattered_pattern_db", pattern_xy_file);
    Utils::to_csv(angles_degrees, pattern_xz_db, "angle_deg", "scattered_pattern_db", pattern_xz_file);

    return 0;
}
