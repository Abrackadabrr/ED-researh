#include "SphereFrequencySweepConfig.hpp"

#include "utility/YamlParser.hpp"

#include <algorithm>
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

constexpr const char *CONFIGURATION_FILE_NAME = "sphere_frequency_sweep.yaml";

[[noreturn]] void invalid_configuration(std::string_view path, std::string_view reason) {
    throw std::invalid_argument("Invalid sphere frequency sweep configuration at '" +
                                std::string{path} + "': " + std::string{reason});
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

Types::index index_value(const YAML::Node &parent,
                         const char *key,
                         std::string_view parent_path,
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

std::string string_value(const YAML::Node &parent, const char *key, std::string_view parent_path) {
    const std::string path = std::string{parent_path} + "." + key;
    const YAML::Node node = required_node(parent, key, parent_path);
    if (!node.IsScalar()) {
        invalid_configuration(path, "expected a string");
    }
    try {
        std::string value = node.as<std::string>();
        if (value.empty()) {
            invalid_configuration(path, "must not be empty");
        }
        return value;
    } catch (const YAML::Exception &) {
        invalid_configuration(path, "expected a string");
    }
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

#ifdef SPHERE_FREQUENCY_SWEEP_SOURCE_CONFIG
    const std::filesystem::path source_candidate = SPHERE_FREQUENCY_SWEEP_SOURCE_CONFIG;
    if (std::filesystem::exists(source_candidate)) {
        return source_candidate;
    }
#endif

    return working_directory_candidate;
}

SphereFrequencySweepConfig configuration_from_yaml(const YAML::Node &root) {
    if (!root || !root.IsMap()) {
        invalid_configuration("root", "expected a map");
    }

    SphereFrequencySweepConfig config;

    const YAML::Node sphere = required_map(root, "sphere", "root");
    const YAML::Node epsilon = required_map(sphere, "relative_permittivity", "sphere");
    config.sphere.epsilon = {scalar_value(epsilon, "real", "sphere.relative_permittivity"),
                             scalar_value(epsilon, "imaginary", "sphere.relative_permittivity")};
    if (std::abs(config.sphere.epsilon) == 0.0) {
        invalid_configuration("sphere.relative_permittivity", "must be non-zero");
    }
    config.sphere.radius = scalar_value(sphere, "radius_m", "sphere");
    if (config.sphere.radius <= 0.0) {
        invalid_configuration("sphere.radius_m", "must be positive");
    }

    const YAML::Node mesh = required_map(root, "mesh", "root");
    config.mesh.min_cells_per_axis = index_value(mesh, "min_cells_per_axis", "mesh");
    config.mesh.cells_per_internal_wavelength =
        scalar_value(mesh, "cells_per_internal_wavelength", "mesh");
    if (config.mesh.cells_per_internal_wavelength <= 0.0) {
        invalid_configuration("mesh.cells_per_internal_wavelength", "must be positive");
    }

    const YAML::Node frequencies = required_map(root, "frequency_sweep_ghz", "root");
    config.frequency_sweep.start_ghz = scalar_value(frequencies, "start", "frequency_sweep_ghz");
    config.frequency_sweep.end_ghz = scalar_value(frequencies, "end", "frequency_sweep_ghz");
    config.frequency_sweep.step_ghz = scalar_value(frequencies, "step", "frequency_sweep_ghz");
    if (config.frequency_sweep.start_ghz <= 0.0) {
        invalid_configuration("frequency_sweep_ghz.start", "must be positive");
    }
    if (config.frequency_sweep.end_ghz < config.frequency_sweep.start_ghz) {
        invalid_configuration("frequency_sweep_ghz.end", "must be greater than or equal to start");
    }
    if (config.frequency_sweep.step_ghz <= 0.0) {
        invalid_configuration("frequency_sweep_ghz.step", "must be positive");
    }
    const Types::scalar frequency_steps =
        (config.frequency_sweep.end_ghz - config.frequency_sweep.start_ghz) /
        config.frequency_sweep.step_ghz;
    if (!std::isfinite(frequency_steps) ||
        frequency_steps > static_cast<Types::scalar>(std::numeric_limits<Types::index>::max() - 1)) {
        invalid_configuration("frequency_sweep_ghz", "the number of frequency points is out of range");
    }
    const Types::scalar rounded_frequency_steps = std::round(frequency_steps);
    if (std::abs(frequency_steps - rounded_frequency_steps) >
        1e-10 * std::max(1.0, std::abs(frequency_steps))) {
        invalid_configuration("frequency_sweep_ghz", "end - start must be an integer multiple of step");
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

    const YAML::Node incident_wave = required_map(root, "incident_wave", "root");
    config.incident_wave.polarization = vector3_value(incident_wave, "polarization", "incident_wave");
    config.incident_wave.direction = vector3_value(incident_wave, "direction", "incident_wave");
    if (config.incident_wave.polarization.squaredNorm() == 0.0) {
        invalid_configuration("incident_wave.polarization", "must be non-zero");
    }
    constexpr Types::scalar unit_vector_tolerance = 1e-10;
    if (std::abs(config.incident_wave.direction.norm() - 1.0) > unit_vector_tolerance) {
        invalid_configuration("incident_wave.direction", "must be a unit vector");
    }
    if (std::abs(config.incident_wave.polarization.dot(config.incident_wave.direction)) >
        unit_vector_tolerance * config.incident_wave.polarization.norm()) {
        invalid_configuration("incident_wave", "polarization must be perpendicular to direction");
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

    const YAML::Node bistatic = required_map(root, "bistatic_diagram", "root");
    config.bistatic_diagram.polar_samples = index_value(bistatic, "polar_samples", "bistatic_diagram");
    config.bistatic_diagram.azimuthal_samples =
        index_value(bistatic, "azimuthal_samples", "bistatic_diagram");
    if (config.bistatic_diagram.polar_samples < 2) {
        invalid_configuration("bistatic_diagram.polar_samples", "must be at least 2");
    }
    if (config.bistatic_diagram.azimuthal_samples < 2) {
        invalid_configuration("bistatic_diagram.azimuthal_samples", "must be at least 2");
    }
    if (config.bistatic_diagram.polar_samples >
        static_cast<Types::index>(std::numeric_limits<Types::integer>::max()) /
            config.bistatic_diagram.azimuthal_samples) {
        invalid_configuration("bistatic_diagram", "the angular grid is too large");
    }

    const YAML::Node output = required_map(root, "output", "root");
    config.output.directory = string_value(output, "directory", "output");
    config.output.summary_file = string_value(output, "summary_file", "output");
    config.output.bistatic_file_prefix = string_value(output, "bistatic_file_prefix", "output");

    return config;
}

} // namespace

SphereFrequencySweepConfig load_sphere_frequency_sweep_config(int argc, char **argv) {
    const std::filesystem::path config_path = configuration_path(argc, argv);
    const YAML::Node yaml = Utility::Yaml::parse_file(config_path);
    SphereFrequencySweepConfig config = configuration_from_yaml(yaml);
    std::cout << "Configuration loaded from " << config_path << std::endl;
    return config;
}

} // namespace Research::Plasmon
