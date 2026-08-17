#pragma once

#include "EMW/types/Types.hpp"

#include <filesystem>
#include <string>

namespace Research::Plasmon {

struct SphereFrequencySweepConfig {
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

    struct BistaticDiagram {
        EMW::Types::index polar_samples{};
        EMW::Types::index azimuthal_samples{};
    } bistatic_diagram;

    struct Output {
        std::filesystem::path directory;
        std::string summary_file;
        std::string bistatic_file_prefix;
    } output;
};

[[nodiscard]] SphereFrequencySweepConfig load_sphere_frequency_sweep_config(int argc, char **argv);

} // namespace Research::Plasmon
