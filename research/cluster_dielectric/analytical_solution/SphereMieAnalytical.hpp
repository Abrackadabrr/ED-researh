#ifndef RESEARCH_CLUSTER_DIELECTRIC_SPHERE_MIE_ANALYTICAL_HPP
#define RESEARCH_CLUSTER_DIELECTRIC_SPHERE_MIE_ANALYTICAL_HPP

#include "mesh/volume_mesh/CubeMeshWithData.hpp"
#include "types/Types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace Research::VolDie::Mie {

struct RSP {
    std::vector<EMW::Types::scalar> hh;
    std::vector<EMW::Types::scalar> vv;
};

struct SolutionComparisonErrors {
    std::size_t compared_cells{};
    EMW::Types::scalar c_norm{};
    EMW::Types::scalar relative_c_norm{};
    EMW::Types::scalar l2_norm{};
    EMW::Types::scalar relative_l2_norm{};
};

/** Calculates the analytical electric field inside a homogeneous dielectric sphere. */
std::vector<EMW::Types::Vector3c>
calculate_field_on_mesh(const EMW::Mesh::VolumeMesh::CubeMeshWithData &mesh,
                        EMW::Types::scalar sphere_radius, EMW::Types::complex_d epsilon,
                        EMW::Types::complex_d wave_number);

/** Calculates the two far-field RSP polarizations from the Mie series. */
RSP calculate_rsp(const std::vector<EMW::Types::scalar> &phis, EMW::Types::scalar sphere_radius,
                  EMW::Types::complex_d epsilon, EMW::Types::complex_d wave_number);

/**
 * Compares the numerical and analytical fields without a phase correction.
 * Only cells whose eight vertices are contained in the sphere participate.
 */
SolutionComparisonErrors
compare_solutions_in_sphere_interior(const EMW::Mesh::VolumeMesh::CubeMeshWithData &mesh,
                                     const std::string &numerical_solution_name,
                                     const std::string &analytical_solution_name,
                                     EMW::Types::scalar sphere_radius);

} // namespace Research::VolDie::Mie

#endif // RESEARCH_CLUSTER_DIELECTRIC_SPHERE_MIE_ANALYTICAL_HPP
