#include "LongPlateDipoleScatteringConfig.hpp"

#include "utility/YamlParser.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace Research::Plasmon {
namespace {

namespace Types = EMW::Types;

constexpr const char *CONFIGURATION_FILE_NAME = "long_plate_dipole_scattering.yaml";

[[noreturn]] void invalid_configuration(std::string_view path, std::string_view reason) {
    throw std::invalid_argument("Invalid long plate dipole scattering configuration at '" + std::string{path} +
                                "': " + std::string{reason});
}

YAML::Node required_node(const YAML::Node &parent, const char *key, std::string_view parent_path) {
    if (!parent || !parent.IsMap()) {
        invalid_configuration(parent_path, "expected a map");
    }

    const YAML::Node node = parent[key];
    if (!node || node.IsNull()) {
        invalid_configuration(std::string{parent_path} + "." + key, "required value is missing");
    }
    return node;
}

YAML::Node required_map(const YAML::Node &parent, const char *key, std::string_view parent_path) {
    const YAML::Node node = required_node(parent, key, parent_path);
    if (!node.IsMap()) {
        invalid_configuration(std::string{parent_path} + "." + key, "expected a map");
    }
    return node;
}

Types::scalar scalar_value(const YAML::Node &parent, const char *key, std::string_view parent_path) {
    const std::string path = std::string{parent_path} + "." + key;
    const YAML::Node node = required_node(parent, key, parent_path);
    if (!node.IsScalar()) {
        invalid_configuration(path, "expected a scalar number");
    }

    try {
        const Types::scalar value = node.as<Types::scalar>();
        if (!std::isfinite(value)) {
            invalid_configuration(path, "number must be finite");
        }
        return value;
    } catch (const YAML::Exception &) {
        invalid_configuration(path, "expected a scalar number");
    }
}

Types::index index_value(const YAML::Node &parent, const char *key, std::string_view parent_path,
                         bool allow_zero = false) {
    const std::string path = std::string{parent_path} + "." + key;
    const YAML::Node node = required_node(parent, key, parent_path);
    if (!node.IsScalar()) {
        invalid_configuration(path, "expected an integer");
    }

    try {
        const long long value = node.as<long long>();
        if (value < 0 || (!allow_zero && value == 0)) {
            invalid_configuration(path, allow_zero ? "must be non-negative" : "must be positive");
        }
        if (static_cast<unsigned long long>(value) >
            static_cast<unsigned long long>(std::numeric_limits<Types::index>::max())) {
            invalid_configuration(path, "integer is out of range");
        }
        return static_cast<Types::index>(value);
    } catch (const YAML::Exception &) {
        invalid_configuration(path, "expected an integer");
    }
}

Types::Vector3d vector3_value(const YAML::Node &parent, const char *key, std::string_view parent_path) {
    const std::string path = std::string{parent_path} + "." + key;
    const YAML::Node node = required_node(parent, key, parent_path);
    if (!node.IsSequence() || node.size() != 3) {
        invalid_configuration(path, "expected a sequence of exactly three numbers");
    }

    Types::Vector3d result;
    for (std::size_t component = 0; component < 3; ++component) {
        try {
            result[static_cast<Types::index>(component)] = node[component].as<Types::scalar>();
        } catch (const YAML::Exception &) {
            invalid_configuration(path + "[" + std::to_string(component) + "]", "expected a scalar number");
        }
        if (!std::isfinite(result[static_cast<Types::index>(component)])) {
            invalid_configuration(path + "[" + std::to_string(component) + "]", "number must be finite");
        }
    }
    return result;
}

std::filesystem::path configuration_path(int argc, char **argv) {
    if (argc > 2) {
        throw std::invalid_argument("Usage: " + std::string{argv[0]} + " [configuration.yaml]");
    }
    if (argc == 2) {
        return argv[1];
    }

    const std::filesystem::path working_directory_candidate = CONFIGURATION_FILE_NAME;
    if (std::filesystem::exists(working_directory_candidate)) {
        return working_directory_candidate;
    }

    const std::filesystem::path executable_candidate =
        std::filesystem::absolute(argv[0]).parent_path() / CONFIGURATION_FILE_NAME;
    if (std::filesystem::exists(executable_candidate)) {
        return executable_candidate;
    }

#ifdef LONG_PLATE_DIPOLE_SCATTERING_SOURCE_CONFIG
    const std::filesystem::path source_candidate = LONG_PLATE_DIPOLE_SCATTERING_SOURCE_CONFIG;
    if (std::filesystem::exists(source_candidate)) {
        return source_candidate;
    }
#endif

    return working_directory_candidate;
}

LongPlateDipoleScatteringConfig configuration_from_yaml(const YAML::Node &root) {
    if (!root || !root.IsMap()) {
        invalid_configuration("root", "expected a map");
    }

    LongPlateDipoleScatteringConfig config;

    config.wavelength = scalar_value(root, "wavelength_m", "root");
    if (config.wavelength <= 0.0) {
        invalid_configuration("root.wavelength_m", "must be positive");
    }

    const YAML::Node plate = required_map(root, "plate", "root");
    const YAML::Node dimensions = required_map(plate, "dimensions_m", "plate");
    config.plate.length = scalar_value(dimensions, "length", "plate.dimensions_m");
    config.plate.width = scalar_value(dimensions, "width", "plate.dimensions_m");
    config.plate.thickness = scalar_value(dimensions, "thickness", "plate.dimensions_m");
    if (config.plate.length <= 0.0 || config.plate.width <= 0.0 || config.plate.thickness <= 0.0) {
        invalid_configuration("plate.dimensions_m", "all dimensions must be positive");
    }
    config.plate.min_corner = vector3_value(plate, "min_corner_m", "plate");

    const YAML::Node epsilon = required_map(plate, "relative_permittivity", "plate");
    config.plate.epsilon = {scalar_value(epsilon, "real", "plate.relative_permittivity"),
                            scalar_value(epsilon, "imaginary", "plate.relative_permittivity")};
    if (std::abs(config.plate.epsilon) == 0.0) {
        invalid_configuration("plate.relative_permittivity", "must be non-zero");
    }

    const YAML::Node mesh = required_map(root, "mesh", "root");
    config.mesh.cells_per_internal_wavelength = scalar_value(mesh, "cells_per_internal_wavelength", "mesh");
    if (config.mesh.cells_per_internal_wavelength <= 0.0) {
        invalid_configuration("mesh.cells_per_internal_wavelength", "must be positive");
    }

    const YAML::Node source = required_map(root, "source", "root");
    config.source.center = vector3_value(source, "center_m", "source");
    config.source.direction = vector3_value(source, "direction", "source");
    if (config.source.direction.squaredNorm() == 0.0) {
        invalid_configuration("source.direction", "must be non-zero");
    }
    config.source.length_wavelengths = scalar_value(source, "length_wavelengths", "source");
    if (config.source.length_wavelengths <= 0.0) {
        invalid_configuration("source.length_wavelengths", "must be positive");
    }
    const YAML::Node current_amplitude = required_map(source, "current_amplitude", "source");
    config.source.current_amplitude = {
        scalar_value(current_amplitude, "real", "source.current_amplitude"),
        scalar_value(current_amplitude, "imaginary", "source.current_amplitude")};
    if (std::abs(config.source.current_amplitude) == 0.0) {
        invalid_configuration("source.current_amplitude", "must be non-zero");
    }

    const Types::Vector3d plate_max =
        config.plate.min_corner + Types::Vector3d{config.plate.length, config.plate.width, config.plate.thickness};
    const bool source_is_inside = (config.source.center.array() >= config.plate.min_corner.array()).all() &&
                                  (config.source.center.array() <= plate_max.array()).all();
    if (source_is_inside) {
        invalid_configuration("source.center_m", "must be outside the plate to avoid the source singularity");
    }

    const YAML::Node integration = required_map(root, "integration", "root");
    config.integration.relative_tolerance = scalar_value(integration, "relative_tolerance", "integration");
    config.integration.absolute_tolerance = scalar_value(integration, "absolute_tolerance", "integration");
    config.integration.level_2d = index_value(integration, "level_2d", "integration", true);
    config.integration.level_3d = index_value(integration, "level_3d", "integration", true);
    config.integration.level_4d = index_value(integration, "level_4d", "integration", true);
    config.integration.level_6d = index_value(integration, "level_6d", "integration", true);
    config.integration.nearness_threshold = index_value(integration, "nearness_threshold", "integration");
    if (config.integration.relative_tolerance <= 0.0) {
        invalid_configuration("integration.relative_tolerance", "must be positive");
    }
    if (config.integration.absolute_tolerance < 0.0) {
        invalid_configuration("integration.absolute_tolerance", "must be non-negative");
    }

    const YAML::Node solver = required_map(root, "solver", "root");
    const YAML::Node gmres = required_map(solver, "gmres", "solver");
    config.gmres.max_iterations = index_value(gmres, "max_iterations", "solver.gmres");
    config.gmres.tolerance = scalar_value(gmres, "tolerance", "solver.gmres");
    config.gmres.restart = index_value(gmres, "restart", "solver.gmres");
    if (config.gmres.tolerance <= 0.0) {
        invalid_configuration("solver.gmres.tolerance", "must be positive");
    }
    if (config.gmres.restart > config.gmres.max_iterations) {
        invalid_configuration("solver.gmres.restart", "must not exceed max_iterations");
    }

    const YAML::Node field_plane = required_map(root, "field_plane", "root");
    config.field_plane.y = scalar_value(field_plane, "y_m", "field_plane");
    config.field_plane.x_min = scalar_value(field_plane, "x_min_m", "field_plane");
    config.field_plane.x_max = scalar_value(field_plane, "x_max_m", "field_plane");
    config.field_plane.z_min = scalar_value(field_plane, "z_min_m", "field_plane");
    config.field_plane.z_max = scalar_value(field_plane, "z_max_m", "field_plane");
    config.field_plane.points_x = index_value(field_plane, "points_x", "field_plane");
    config.field_plane.points_z = index_value(field_plane, "points_z", "field_plane");
    if (config.field_plane.x_max <= config.field_plane.x_min) {
        invalid_configuration("field_plane", "x_max_m must be greater than x_min_m");
    }
    if (config.field_plane.z_max <= config.field_plane.z_min) {
        invalid_configuration("field_plane", "z_max_m must be greater than z_min_m");
    }
    if (config.field_plane.points_x < 2 || config.field_plane.points_z < 2) {
        invalid_configuration("field_plane", "points_x and points_z must be at least 2");
    }

    const Types::index angular_samples = index_value(root, "angular_samples", "root");
    if (static_cast<unsigned long long>(angular_samples) >
        static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        invalid_configuration("root.angular_samples", "integer is out of range");
    }
    config.angular_samples = static_cast<int>(angular_samples);

    const YAML::Node output = required_map(root, "output", "root");
    const YAML::Node output_directory = required_node(output, "directory", "output");
    if (!output_directory.IsScalar()) {
        invalid_configuration("output.directory", "expected a path string");
    }
    try {
        config.output_directory = output_directory.as<std::string>();
    } catch (const YAML::Exception &) {
        invalid_configuration("output.directory", "expected a path string");
    }
    if (config.output_directory.empty()) {
        invalid_configuration("output.directory", "must not be empty");
    }

    return config;
}

} // namespace

LongPlateDipoleScatteringConfig load_long_plate_dipole_scattering_config(int argc, char **argv) {
    const std::filesystem::path config_path = configuration_path(argc, argv);
    const YAML::Node yaml = Utility::Yaml::parse_file(config_path);
    LongPlateDipoleScatteringConfig config = configuration_from_yaml(yaml);
    std::cout << "Configuration loaded from " << config_path << std::endl;
    return config;
}

} // namespace Research::Plasmon
