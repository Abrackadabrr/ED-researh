#include "PillarArrayScatteringConfig.hpp"

#include "utility/YamlParser.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace Research::Plasmon {
namespace {

namespace Types = EMW::Types;

constexpr const char *CONFIGURATION_FILE_NAME = "pillar_array_scattering.yaml";

[[noreturn]] void invalid_configuration(std::string_view path, std::string_view reason) {
    throw std::invalid_argument("Invalid pillar array scattering configuration at '" + std::string{path} +
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

Types::Vector2d vector2_value(const YAML::Node &parent, const char *key, std::string_view parent_path) {
    const std::string path = std::string{parent_path} + "." + key;
    const YAML::Node node = required_node(parent, key, parent_path);
    if (!node.IsSequence() || node.size() != 2) {
        invalid_configuration(path, "expected a sequence of exactly two numbers");
    }

    Types::Vector2d result;
    for (std::size_t component = 0; component < 2; ++component) {
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

std::array<Types::index, 2> index_pair(const YAML::Node &node, std::string_view path, bool allow_zero) {
    if (!node.IsSequence() || node.size() != 2) {
        invalid_configuration(path, "expected a sequence of exactly two integers");
    }

    std::array<Types::index, 2> result{};
    for (std::size_t component = 0; component < 2; ++component) {
        try {
            const long long value = node[component].as<long long>();
            if (value < 0 || (!allow_zero && value == 0)) {
                invalid_configuration(std::string{path} + "[" + std::to_string(component) + "]",
                                      allow_zero ? "must be non-negative" : "must be positive");
            }
            result[component] = static_cast<Types::index>(value);
        } catch (const YAML::Exception &) {
            invalid_configuration(std::string{path} + "[" + std::to_string(component) + "]",
                                  "expected an integer");
        }
    }
    return result;
}

Types::complex_d complex_value(const YAML::Node &parent, const char *key, std::string_view parent_path) {
    const YAML::Node value = required_map(parent, key, parent_path);
    const std::string path = std::string{parent_path} + "." + key;
    return {scalar_value(value, "real", path), scalar_value(value, "imaginary", path)};
}

std::filesystem::path path_value(const YAML::Node &parent, const char *key, std::string_view parent_path) {
    const std::string path = std::string{parent_path} + "." + key;
    const YAML::Node node = required_node(parent, key, parent_path);
    if (!node.IsScalar()) {
        invalid_configuration(path, "expected a path string");
    }
    try {
        const std::filesystem::path result = node.as<std::string>();
        if (result.empty()) {
            invalid_configuration(path, "must not be empty");
        }
        return result;
    } catch (const YAML::Exception &) {
        invalid_configuration(path, "expected a path string");
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

#ifdef PILLAR_ARRAY_SCATTERING_SOURCE_CONFIG
    const std::filesystem::path source_candidate = PILLAR_ARRAY_SCATTERING_SOURCE_CONFIG;
    if (std::filesystem::exists(source_candidate)) {
        return source_candidate;
    }
#endif

    return working_directory_candidate;
}

PillarArrayGeometryConfig geometry_from_yaml(const YAML::Node &root) {
    if (!root || !root.IsMap()) {
        invalid_configuration("geometry", "expected a map");
    }

    PillarArrayGeometryConfig geometry;
    const YAML::Node lattice = required_map(root, "lattice", "geometry");
    geometry.lattice.min_corner_nm = vector3_value(lattice, "min_corner_nm", "geometry.lattice");
    geometry.lattice.cells = index_pair(required_node(lattice, "cells", "geometry.lattice"),
                                        "geometry.lattice.cells", false);
    geometry.lattice.cell_size_nm = vector2_value(lattice, "cell_size_nm", "geometry.lattice");
    if ((geometry.lattice.cell_size_nm.array() <= 0.0).any()) {
        invalid_configuration("geometry.lattice.cell_size_nm", "both sizes must be positive");
    }

    const YAML::Node substrate = required_map(root, "substrate", "geometry");
    geometry.substrate_thickness_nm = scalar_value(substrate, "thickness_nm", "geometry.substrate");
    if (geometry.substrate_thickness_nm <= 0.0) {
        invalid_configuration("geometry.substrate.thickness_nm", "must be positive");
    }

    const YAML::Node pillar = required_map(root, "pillar", "geometry");
    geometry.pillar_size_nm = vector3_value(pillar, "size_nm", "geometry.pillar");
    if ((geometry.pillar_size_nm.array() <= 0.0).any()) {
        invalid_configuration("geometry.pillar.size_nm", "all sizes must be positive");
    }
    if (geometry.pillar_size_nm.x() > geometry.lattice.cell_size_nm.x() ||
        geometry.pillar_size_nm.y() > geometry.lattice.cell_size_nm.y()) {
        invalid_configuration("geometry.pillar.size_nm", "pillar cross-section must fit into a lattice cell");
    }

    const YAML::Node centers = required_node(pillar, "centers", "geometry.pillar");
    if (!centers.IsSequence() || centers.size() == 0) {
        invalid_configuration("geometry.pillar.centers", "expected a non-empty sequence of integer pairs");
    }
    std::set<std::array<Types::index, 2>> unique_centers;
    geometry.pillar_centers.reserve(centers.size());
    for (std::size_t item = 0; item < centers.size(); ++item) {
        const std::string path = "geometry.pillar.centers[" + std::to_string(item) + "]";
        const auto center = index_pair(centers[item], path, true);
        if (center[0] >= geometry.lattice.cells[0] || center[1] >= geometry.lattice.cells[1]) {
            invalid_configuration(path, "coordinate lies outside the finite lattice");
        }
        if (!unique_centers.insert(center).second) {
            invalid_configuration(path, "duplicate pillar coordinate");
        }
        geometry.pillar_centers.push_back(center);
    }
    return geometry;
}

bool is_integer_multiple(Types::scalar value, Types::scalar step) {
    const Types::scalar ratio = value / step;
    return std::abs(ratio - std::round(ratio)) <= 1.0e-10 * std::max(1.0, std::abs(ratio));
}

void validate_geometry_against_mesh(const PillarArrayScatteringConfig &config) {
    const auto &geometry = config.geometry;
    const Types::scalar cell_size_scale = std::max({1.0, geometry.lattice.cell_size_nm.x(),
                                                    geometry.lattice.cell_size_nm.y()});
    if (std::abs(geometry.lattice.cell_size_nm.x() - geometry.lattice.cell_size_nm.y()) >
        1.0e-12 * cell_size_scale) {
        invalid_configuration("geometry.lattice.cell_size_nm",
                              "x and y sizes must be equal because the volume mesh must be cubic");
    }

    const Types::scalar step_nm =
        geometry.lattice.cell_size_nm.x() / static_cast<Types::scalar>(config.mesh.cells_per_geometry_cell_side);
    const std::array<std::pair<Types::scalar, std::string_view>, 5> aligned_lengths{{
        {geometry.substrate_thickness_nm, "geometry.substrate.thickness_nm"},
        {geometry.pillar_size_nm.x(), "geometry.pillar.size_nm[0]"},
        {geometry.pillar_size_nm.y(), "geometry.pillar.size_nm[1]"},
        {geometry.pillar_size_nm.z(), "geometry.pillar.size_nm[2]"},
        {(geometry.lattice.cell_size_nm.x() - geometry.pillar_size_nm.x()) / 2.0,
         "geometry.pillar x margin"},
    }};
    for (const auto &[length, path] : aligned_lengths) {
        if (!is_integer_multiple(length, step_nm)) {
            invalid_configuration(path, "must be an integer multiple of the resulting volume-mesh step");
        }
    }
    const Types::scalar y_margin_nm =
        (geometry.lattice.cell_size_nm.y() - geometry.pillar_size_nm.y()) / 2.0;
    if (!is_integer_multiple(y_margin_nm, step_nm)) {
        invalid_configuration("geometry.pillar y margin",
                              "must be an integer multiple of the resulting volume-mesh step");
    }
}

PillarArrayScatteringConfig configuration_from_yaml(const YAML::Node &root,
                                                     const std::filesystem::path &config_path) {
    if (!root || !root.IsMap()) {
        invalid_configuration("root", "expected a map");
    }

    PillarArrayScatteringConfig config;
    config.wavelength_nm = scalar_value(root, "wavelength_nm", "root");
    if (config.wavelength_nm <= 0.0) {
        invalid_configuration("root.wavelength_nm", "must be positive");
    }

    config.geometry_file = path_value(root, "geometry_file", "root");
    if (config.geometry_file.is_relative()) {
        config.geometry_file = config_path.parent_path() / config.geometry_file;
    }
    config.geometry = geometry_from_yaml(Utility::Yaml::parse_file(config.geometry_file));

    const YAML::Node materials = required_map(root, "materials", "root");
    config.materials.substrate_epsilon =
        complex_value(materials, "substrate_relative_permittivity", "materials");
    config.materials.pillar_epsilon = complex_value(materials, "pillar_relative_permittivity", "materials");

    const YAML::Node mesh = required_map(root, "mesh", "root");
    config.mesh.cells_per_geometry_cell_side =
        index_value(mesh, "cells_per_geometry_cell_side", "mesh");

    const YAML::Node incident_wave = required_map(root, "incident_wave", "root");
    config.incident_wave.direction = vector3_value(incident_wave, "direction", "incident_wave");
    config.incident_wave.polarization = vector3_value(incident_wave, "polarization", "incident_wave");
    constexpr Types::scalar vector_tolerance = 1.0e-10;
    if (std::abs(config.incident_wave.direction.norm() - 1.0) > vector_tolerance) {
        invalid_configuration("incident_wave.direction", "must be a unit vector");
    }
    if (config.incident_wave.polarization.squaredNorm() == 0.0) {
        invalid_configuration("incident_wave.polarization", "must be non-zero");
    }
    if (std::abs(config.incident_wave.direction.dot(config.incident_wave.polarization)) >
        vector_tolerance * config.incident_wave.polarization.norm()) {
        invalid_configuration("incident_wave", "polarization must be perpendicular to direction");
    }
    const Types::Vector3d substrate_normal{0.0, 0.0, 1.0};
    if (config.incident_wave.direction.cross(substrate_normal).norm() <= vector_tolerance) {
        invalid_configuration("incident_wave.direction",
                              "must be oblique to the substrate normal to define an incidence plane");
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

    const YAML::Node bistatic_rcs = required_map(root, "bistatic_rcs", "root");
    config.bistatic_rcs.phi_min_deg = scalar_value(bistatic_rcs, "phi_min_deg", "bistatic_rcs");
    config.bistatic_rcs.phi_max_deg = scalar_value(bistatic_rcs, "phi_max_deg", "bistatic_rcs");
    config.bistatic_rcs.samples = index_value(bistatic_rcs, "samples", "bistatic_rcs");
    if (config.bistatic_rcs.phi_max_deg <= config.bistatic_rcs.phi_min_deg) {
        invalid_configuration("bistatic_rcs", "phi_max_deg must be greater than phi_min_deg");
    }
    if (config.bistatic_rcs.samples < 2) {
        invalid_configuration("bistatic_rcs.samples", "must be at least 2");
    }

    const YAML::Node output = required_map(root, "output", "root");
    config.output_directory = path_value(output, "directory", "output");

    validate_geometry_against_mesh(config);
    return config;
}

} // namespace

PillarArrayScatteringConfig load_pillar_array_scattering_config(int argc, char **argv) {
    const std::filesystem::path config_path = configuration_path(argc, argv);
    const YAML::Node yaml = Utility::Yaml::parse_file(config_path);
    PillarArrayScatteringConfig config = configuration_from_yaml(yaml, config_path);
    std::cout << "Configuration loaded from " << config_path << '\n'
              << "Geometry loaded from " << config.geometry_file << std::endl;
    return config;
}

} // namespace Research::Plasmon
