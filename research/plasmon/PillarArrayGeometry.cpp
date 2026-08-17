#include "PillarArrayGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Research::Plasmon {
namespace {

namespace Types = EMW::Types;

bool in_half_open_interval(Types::scalar value, Types::scalar lower, Types::scalar upper) {
    return value >= lower && value < upper;
}

bool point_in_box(const Types::point_t &point, const Types::point_t &lower, const Types::point_t &upper) {
    return in_half_open_interval(point.x(), lower.x(), upper.x()) &&
           in_half_open_interval(point.y(), lower.y(), upper.y()) &&
           in_half_open_interval(point.z(), lower.z(), upper.z());
}

bool approximately_equal(Types::scalar lhs, Types::scalar rhs) {
    return std::abs(lhs - rhs) <= 1.0e-11 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

} // namespace

PillarArrayMeshDescription make_pillar_array_mesh_description(const PillarArrayScatteringConfig &config) {
    const auto &geometry = config.geometry;
    // Scale invariance: the physical wavelength in nanometers is mapped to exactly one model meter.
    const Types::scalar scale = 1.0 / config.wavelength_nm;
    PillarArrayMeshDescription description;
    description.min_corner = scale * geometry.lattice.min_corner_nm;
    description.size_x = scale * geometry.lattice.cells[0] * geometry.lattice.cell_size_nm.x();
    description.size_y = scale * geometry.lattice.cells[1] * geometry.lattice.cell_size_nm.y();
    description.size_z = scale * (geometry.substrate_thickness_nm + geometry.pillar_size_nm.z());
    description.step = scale * geometry.lattice.cell_size_nm.x() /
                       static_cast<Types::scalar>(config.mesh.cells_per_geometry_cell_side);
    description.cells_x = geometry.lattice.cells[0] * config.mesh.cells_per_geometry_cell_side;
    description.cells_y = geometry.lattice.cells[1] * config.mesh.cells_per_geometry_cell_side;
    description.cells_z = static_cast<Types::index>(std::llround(description.size_z / description.step));
    return description;
}

Types::complex_d relative_permittivity_at(const Types::point_t &point,
                                          const PillarArrayScatteringConfig &config) {
    const auto &geometry = config.geometry;
    const Types::point_t point_nm = point * config.wavelength_nm;
    const Types::scalar pillar_base_nm =
        geometry.lattice.min_corner_nm.z() + geometry.substrate_thickness_nm;

    if (in_half_open_interval(point_nm.z(), pillar_base_nm,
                              pillar_base_nm + geometry.pillar_size_nm.z())) {
        for (const auto &coordinate : geometry.pillar_centers) {
            const Types::scalar center_x_nm = geometry.lattice.min_corner_nm.x() +
                                              (static_cast<Types::scalar>(coordinate[0]) + 0.5) *
                                                  geometry.lattice.cell_size_nm.x();
            const Types::scalar center_y_nm = geometry.lattice.min_corner_nm.y() +
                                              (static_cast<Types::scalar>(coordinate[1]) + 0.5) *
                                                  geometry.lattice.cell_size_nm.y();
            const Types::point_t lower_nm{center_x_nm - geometry.pillar_size_nm.x() / 2.0,
                                          center_y_nm - geometry.pillar_size_nm.y() / 2.0,
                                          pillar_base_nm};
            const Types::point_t upper_nm{center_x_nm + geometry.pillar_size_nm.x() / 2.0,
                                          center_y_nm + geometry.pillar_size_nm.y() / 2.0,
                                          pillar_base_nm + geometry.pillar_size_nm.z()};
            if (point_in_box(point_nm, lower_nm, upper_nm)) {
                return config.materials.pillar_epsilon;
            }
        }
    }

    const Types::point_t substrate_upper_nm{
        geometry.lattice.min_corner_nm.x() + geometry.lattice.cells[0] * geometry.lattice.cell_size_nm.x(),
        geometry.lattice.min_corner_nm.y() + geometry.lattice.cells[1] * geometry.lattice.cell_size_nm.y(),
        pillar_base_nm};
    if (point_in_box(point_nm, geometry.lattice.min_corner_nm, substrate_upper_nm)) {
        return config.materials.substrate_epsilon;
    }
    return {1.0, 0.0};
}

std::vector<Types::complex_d> make_pillar_array_permittivity(
    const EMW::Mesh::VolumeMesh::CubeMesh &mesh, const PillarArrayScatteringConfig &config) {
    const PillarArrayMeshDescription expected = make_pillar_array_mesh_description(config);
    if (mesh.nCubesX() != expected.cells_x || mesh.nCubesY() != expected.cells_y ||
        mesh.nCubesZ() != expected.cells_z || !approximately_equal(mesh.dx(), expected.step) ||
        !approximately_equal(mesh.dy(), expected.step) || !approximately_equal(mesh.dz(), expected.step) ||
        mesh.getNodes().empty() || !mesh.getNodes().front().isApprox(expected.min_corner, 1.0e-11)) {
        throw std::invalid_argument("Pillar-array mesh does not match the geometry discretization");
    }

    std::vector<Types::complex_d> permittivity(mesh.getCells().size(), Types::complex_d{1.0, 0.0});
    for (Types::index z = 0; z < expected.cells_z; ++z) {
        for (Types::index y = 0; y < expected.cells_y; ++y) {
            for (Types::index x = 0; x < expected.cells_x; ++x) {
                const Types::index cell = mesh.cube_idx(x, y, z);
                permittivity[cell] = relative_permittivity_at(mesh.getCells()[cell].center_, config);
            }
        }
    }
    return permittivity;
}

Types::VectorXc make_pillar_array_epsilon_minus_one(const std::vector<Types::complex_d> &permittivity) {
    Types::VectorXc epsilon_minus_one = Types::VectorXc::Zero(3 * permittivity.size());
    for (Types::index cell = 0; cell < permittivity.size(); ++cell) {
        const Types::complex_d contrast = permittivity[cell] - Types::complex_d{1.0, 0.0};
        epsilon_minus_one.segment<3>(3 * cell).setConstant(contrast);
    }
    return epsilon_minus_one;
}

} // namespace Research::Plasmon
