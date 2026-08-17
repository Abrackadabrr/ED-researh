#include "SpherePlasmonRegimeConfig.hpp"

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

constexpr const char *CONFIGURATION_FILE_NAME = "sphere_plasmon_regime.yaml";

[[noreturn]] void invalid_configuration(std::string_view path, std::string_view reason) {
    throw std::invalid_argument("Invalid sphere plasmon regime configuration at '" + std::string{path} +
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

#ifdef SPHERE_PLASMON_REGIME_SOURCE_CONFIG
    const std::filesystem::path source_candidate = SPHERE_PLASMON_REGIME_SOURCE_CONFIG;
    if (std::filesystem::exists(source_candidate)) {
        return source_candidate;
    }
#endif

    return working_directory_candidate;
}

SpherePlasmonRegimeConfig configuration_from_yaml(const YAML::Node &root) {
    if (!root || !root.IsMap()) {
        invalid_configuration("root", "expected a map");
    }

    SpherePlasmonRegimeConfig config;

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
    const YAML::Node points_per_axis = required_node(mesh, "points_per_axis", "mesh");
    if (!points_per_axis.IsSequence() || points_per_axis.size() == 0) {
        invalid_configuration("mesh.points_per_axis", "expected a non-empty integer sequence");
    }
    config.mesh.points_per_axis.reserve(points_per_axis.size());
    for (std::size_t item = 0; item < points_per_axis.size(); ++item) {
        const std::string path = "mesh.points_per_axis[" + std::to_string(item) + "]";
        try {
            const long long value = points_per_axis[item].as<long long>();
            if (value < 2) {
                invalid_configuration(path, "must be at least 2");
            }
            if (static_cast<unsigned long long>(value) >
                static_cast<unsigned long long>(std::numeric_limits<Types::index>::max())) {
                invalid_configuration(path, "integer is out of range");
            }
            config.mesh.points_per_axis.push_back(static_cast<Types::index>(value));
        } catch (const YAML::Exception &) {
            invalid_configuration(path, "expected an integer");
        }
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

    config.frequency_ghz = scalar_value(root, "frequency_ghz", "root");
    if (config.frequency_ghz <= 0.0) {
        invalid_configuration("root.frequency_ghz", "must be positive");
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
    config.analytical_rsp_db_offset = scalar_value(output, "analytical_rsp_db_offset", "output");

    return config;
}

} // namespace

SpherePlasmonRegimeConfig load_sphere_plasmon_regime_config(int argc, char **argv) {
    const std::filesystem::path config_path = configuration_path(argc, argv);
    const YAML::Node yaml = Utility::Yaml::parse_file(config_path);
    SpherePlasmonRegimeConfig config = configuration_from_yaml(yaml);
    std::cout << "Configuration loaded from " << config_path << std::endl;
    return config;
}

} // namespace Research::Plasmon
