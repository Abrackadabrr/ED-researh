#pragma once

#include "EMW/types/Types.hpp"

#include <filesystem>

#include <yaml-cpp/yaml.h>

namespace Research::ClusterDielectric {

struct ClusterSphereExtinctionConfig {
    struct Sphere {
        EMW::Types::complex_d epsilon{};
        EMW::Types::scalar radius{};
    } sphere;

    struct Mesh {
        EMW::Types::index min_cells_per_axis{};
        EMW::Types::scalar cells_per_internal_wavelength{};
    } mesh;

    struct FrequencySweep {
        EMW::Types::scalar start_ghz{};
        EMW::Types::scalar end_ghz{};
        EMW::Types::scalar step_ghz{};
    } frequency_sweep;

    struct Integration {
        EMW::Types::scalar relative_tolerance{};
        EMW::Types::scalar absolute_tolerance{};
        EMW::Types::index level_2d{};
        EMW::Types::index level_3d{};
        EMW::Types::index level_4d{};
        EMW::Types::index level_6d{};
        EMW::Types::index nearness_threshold{};
    } integration;

    struct IncidentWave {
        EMW::Types::Vector3d polarization{};
        EMW::Types::Vector3d direction{};
    } incident_wave;

    struct Gmres {
        EMW::Types::index max_iterations{};
        EMW::Types::scalar tolerance{};
        EMW::Types::index restart{};
    } gmres;

    std::filesystem::path output_file;
};

// Checks required keys, YAML value types, numeric ranges and cross-field constraints.
// Throws std::invalid_argument with the failing YAML path on an invalid document.
void validate_cluster_sphere_extinction_yaml(const YAML::Node &root);

// Validation is always performed before the typed configuration is returned.
[[nodiscard]] ClusterSphereExtinctionConfig
cluster_sphere_extinction_config_from_yaml(const YAML::Node &root);

} // namespace Research::ClusterDielectric
