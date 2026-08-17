#pragma once

#include "EMW/types/Types.hpp"

#include <filesystem>

namespace Research::Plasmon {

struct LongPlateDipoleScatteringConfig {
    EMW::Types::scalar wavelength{};

    struct Plate {
        EMW::Types::scalar length{};
        EMW::Types::scalar width{};
        EMW::Types::scalar thickness{};
        EMW::Types::Vector3d min_corner{};
        EMW::Types::complex_d epsilon{};
    } plate;

    struct Mesh {
        EMW::Types::scalar cells_per_internal_wavelength{};
    } mesh;

    struct Source {
        EMW::Types::Vector3d center{};
        EMW::Types::Vector3d direction{};
        EMW::Types::scalar length_wavelengths{};
        EMW::Types::complex_d current_amplitude{};
    } source;

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

    struct FieldPlane {
        EMW::Types::scalar y{};
        EMW::Types::scalar x_min{};
        EMW::Types::scalar x_max{};
        EMW::Types::scalar z_min{};
        EMW::Types::scalar z_max{};
        EMW::Types::index points_x{};
        EMW::Types::index points_z{};
    } field_plane;

    int angular_samples{};
    std::filesystem::path output_directory;
};

// Resolves the YAML path, reads the document, validates it and returns a typed configuration.
[[nodiscard]] LongPlateDipoleScatteringConfig load_long_plate_dipole_scattering_config(int argc, char **argv);

} // namespace Research::Plasmon
