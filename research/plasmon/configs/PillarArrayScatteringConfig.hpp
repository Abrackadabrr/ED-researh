#pragma once

#include "EMW/types/Types.hpp"

#include <array>
#include <filesystem>
#include <vector>

namespace Research::Plasmon {

struct PillarArrayGeometryConfig {
    struct Lattice {
        EMW::Types::point_t min_corner_nm{};
        std::array<EMW::Types::index, 2> cells{};
        EMW::Types::Vector2d cell_size_nm{};
    } lattice;

    EMW::Types::scalar substrate_thickness_nm{};
    EMW::Types::Vector3d pillar_size_nm{};
    std::vector<std::array<EMW::Types::index, 2>> pillar_centers;
};

struct PillarArrayScatteringConfig {
    EMW::Types::scalar wavelength_nm{};
    std::filesystem::path geometry_file;
    PillarArrayGeometryConfig geometry;

    struct Materials {
        EMW::Types::complex_d substrate_epsilon{};
        EMW::Types::complex_d pillar_epsilon{};
    } materials;

    struct Mesh {
        EMW::Types::index cells_per_geometry_cell_side{};
    } mesh;

    struct IncidentWave {
        EMW::Types::Vector3d direction{};
        EMW::Types::Vector3d polarization{};
    } incident_wave;

    struct Integration {
        EMW::Types::scalar relative_tolerance{};
        EMW::Types::scalar absolute_tolerance{};
        EMW::Types::index level_2d{};
        EMW::Types::index level_3d{};
        EMW::Types::index level_4d{};
        EMW::Types::index level_6d{};
        EMW::Types::index nearness_threshold{};
    } integration;

    struct Gmres {
        EMW::Types::index max_iterations{};
        EMW::Types::scalar tolerance{};
        EMW::Types::index restart{};
    } gmres;

    struct BistaticRcs {
        EMW::Types::scalar phi_min_deg{};
        EMW::Types::scalar phi_max_deg{};
        EMW::Types::index samples{};
    } bistatic_rcs;

    std::filesystem::path output_directory;
};

// Reads and validates both the experiment YAML and its referenced geometry YAML.
[[nodiscard]] PillarArrayScatteringConfig load_pillar_array_scattering_config(int argc, char **argv);

} // namespace Research::Plasmon
