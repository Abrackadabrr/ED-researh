#pragma once

#include "EMW/types/Types.hpp"

#include <filesystem>
#include <vector>

namespace Research::Plasmon {

struct SpherePlasmonRegimeConfig {
    struct Sphere {
        EMW::Types::complex_d epsilon{};
        EMW::Types::scalar radius{};
    } sphere;

    struct Mesh {
        std::vector<EMW::Types::index> points_per_axis;
    } mesh;

    struct Integration {
        EMW::Types::scalar relative_tolerance{};
        EMW::Types::scalar absolute_tolerance{};
        EMW::Types::index level_2d{};
        EMW::Types::index level_3d{};
        EMW::Types::index level_4d{};
        EMW::Types::index level_6d{};
        EMW::Types::index nearness_threshold{};
    } integration;

    EMW::Types::scalar frequency_ghz{};

    struct IncidentWave {
        EMW::Types::Vector3d polarization{};
        EMW::Types::Vector3d direction{};
    } incident_wave;

    struct Gmres {
        EMW::Types::index max_iterations{};
        EMW::Types::scalar tolerance{};
        EMW::Types::index restart{};
    } gmres;

    int angular_samples{};
    EMW::Types::scalar analytical_rsp_db_offset{};
    std::filesystem::path output_directory;
};

// Resolves the YAML path, reads the document, validates it and returns a typed configuration.
[[nodiscard]] SpherePlasmonRegimeConfig load_sphere_plasmon_regime_config(int argc, char **argv);

} // namespace Research::Plasmon
