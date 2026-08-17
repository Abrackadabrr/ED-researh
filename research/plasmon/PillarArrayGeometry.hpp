#pragma once

#include "configs/PillarArrayScatteringConfig.hpp"

#include "EMW/mesh/volume_mesh/CubeMesh.hpp"
#include "EMW/types/Types.hpp"

#include <vector>

namespace Research::Plasmon {

struct PillarArrayMeshDescription {
    EMW::Types::point_t min_corner{};
    EMW::Types::scalar size_x{};
    EMW::Types::scalar size_y{};
    EMW::Types::scalar size_z{};
    EMW::Types::scalar step{};
    EMW::Types::index cells_x{};
    EMW::Types::index cells_y{};
    EMW::Types::index cells_z{};
};

[[nodiscard]] PillarArrayMeshDescription
make_pillar_array_mesh_description(const PillarArrayScatteringConfig &config);

[[nodiscard]] EMW::Types::complex_d relative_permittivity_at(
    const EMW::Types::point_t &point, const PillarArrayScatteringConfig &config);

// Values are written in CubeMesh::cube_idx order: x is the fastest index, then y, then z.
[[nodiscard]] std::vector<EMW::Types::complex_d>
make_pillar_array_permittivity(const EMW::Mesh::VolumeMesh::CubeMesh &mesh,
                               const PillarArrayScatteringConfig &config);

// Repeats every cell contrast for its consecutive x, y and z field components.
[[nodiscard]] EMW::Types::VectorXc
make_pillar_array_epsilon_minus_one(const std::vector<EMW::Types::complex_d> &permittivity);

} // namespace Research::Plasmon
