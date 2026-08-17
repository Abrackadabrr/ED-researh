#include "SphereMieAnalytical.hpp"

#include "nmie.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace Research::VolDie::Mie {

std::vector<EMW::Types::Vector3c>
calculate_field_on_mesh(const EMW::Mesh::VolumeMesh::CubeMeshWithData &mesh,
                        EMW::Types::scalar sphere_radius, EMW::Types::complex_d epsilon,
                        EMW::Types::complex_d wave_number) {
    const auto &cells = mesh.getCells();
    const std::size_t total_points = cells.size();
    const EMW::Types::scalar k0 = wave_number.real();

    // scattnlay uses dimensionless coordinates k0*r and the size parameter k0*R.
    std::vector<double> layer_size{k0 * sphere_radius};
    // The Mie solver expects a refractive index, rather than dielectric permittivity.
    std::vector<std::complex<double>> refractive_index{std::sqrt(epsilon)};
    std::vector<double> x(total_points);
    std::vector<double> y(total_points);
    std::vector<double> z(total_points);

    for (std::size_t index = 0; index < total_points; ++index) {
        const auto &center = cells[index].center_;
        x[index] = k0 * center.x();
        y[index] = k0 * center.y();
        z[index] = k0 * center.z();
    }

    std::vector<std::vector<std::complex<double>>> electric_field(
        total_points, std::vector<std::complex<double>>(3));
    std::vector<std::vector<std::complex<double>>> magnetic_field(
        total_points, std::vector<std::complex<double>>(3));

    // nField returns the full Cartesian field: incident plus scattered.
    const int nmax = nmie::nField(1, -1, layer_size, refractive_index, -1, nmie::Modes::kAll,
                                  nmie::Modes::kAll, static_cast<unsigned int>(total_points), x, y, z,
                                  electric_field, magnetic_field);
    std::cout << "Analytical solution nmax = " << nmax << std::endl;

    std::vector<EMW::Types::Vector3c> field_on_mesh;
    field_on_mesh.reserve(total_points);
    for (std::size_t index = 0; index < total_points; ++index) {
        if (cells[index].center_.norm() < sphere_radius) {
            field_on_mesh.emplace_back(electric_field[index][0], electric_field[index][1],
                                       electric_field[index][2]);
        } else {
            field_on_mesh.emplace_back(EMW::Types::Vector3c::Zero());
        }
    }
    return field_on_mesh;
}

RSP calculate_rsp(const std::vector<EMW::Types::scalar> &phis, EMW::Types::scalar sphere_radius,
                  EMW::Types::complex_d epsilon, EMW::Types::complex_d wave_number) {
    const EMW::Types::scalar k0 = wave_number.real();
    std::vector<double> layer_size{k0 * sphere_radius};
    std::vector<std::complex<double>> refractive_index{std::sqrt(epsilon)};
    std::vector<double> theta{phis.begin(), phis.end()};

    double qext = 0.0;
    double qsca = 0.0;
    double qabs = 0.0;
    double qbk = 0.0;
    double qpr = 0.0;
    double asymmetry = 0.0;
    double albedo = 0.0;
    std::vector<std::complex<double>> s1;
    std::vector<std::complex<double>> s2;

    const int nmax = nmie::nMie(1, layer_size, refractive_index,
                                static_cast<unsigned int>(theta.size()), theta, &qext, &qsca, &qabs,
                                &qbk, &qpr, &asymmetry, &albedo, s1, s2);
    std::cout << "Mie RSP nmax = " << nmax << std::endl;

    RSP rsp;
    rsp.hh.resize(phis.size());
    rsp.vv.resize(phis.size());
    const EMW::Types::scalar scale = 4.0 * M_PI / (k0 * k0);
    for (std::size_t index = 0; index < phis.size(); ++index) {
        rsp.vv[index] = scale * std::norm(s1[index]);
        rsp.hh[index] = scale * std::norm(s2[index]);
    }
    return rsp;
}

SolutionComparisonErrors
compare_solutions_in_sphere_interior(const EMW::Mesh::VolumeMesh::CubeMeshWithData &mesh,
                                     const std::string &numerical_solution_name,
                                     const std::string &analytical_solution_name,
                                     EMW::Types::scalar sphere_radius) {
    const auto &cells = mesh.getCells();
    const auto &numerical_solution = mesh.getVectorData(numerical_solution_name);
    const auto &analytical_solution = mesh.getVectorData(analytical_solution_name);
    if (numerical_solution.size() != cells.size() || analytical_solution.size() != cells.size()) {
        throw std::invalid_argument("Solution size does not match the number of mesh cells");
    }

    const EMW::Types::scalar sphere_radius_squared = sphere_radius * sphere_radius;
    EMW::Types::scalar max_error = 0;
    EMW::Types::scalar max_analytical_solution = 0;
    EMW::Types::scalar squared_l2_error = 0;
    EMW::Types::scalar squared_analytical_l2_norm = 0;
    std::size_t compared_cells = 0;

    for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
        const auto &cell = cells[cell_index];
        const bool cell_is_fully_inside =
            std::ranges::all_of(cell.vertexes_, [sphere_radius_squared](const EMW::Types::point_t &vertex) {
                return vertex.squaredNorm() <= sphere_radius_squared;
            });
        if (!cell_is_fully_inside) {
            continue;
        }

        const EMW::Types::Vector3c difference =
            numerical_solution[cell_index] - analytical_solution[cell_index];
        const EMW::Types::scalar pointwise_error = difference.norm();
        const EMW::Types::scalar analytical_pointwise_norm =
            analytical_solution[cell_index].norm();

        max_error = std::max(max_error, pointwise_error);
        max_analytical_solution = std::max(max_analytical_solution, analytical_pointwise_norm);
        squared_l2_error += cell.volume_ * difference.squaredNorm();
        squared_analytical_l2_norm +=
            cell.volume_ * analytical_solution[cell_index].squaredNorm();
        ++compared_cells;
    }

    if (compared_cells == 0) {
        throw std::runtime_error("There are no mesh cells fully contained in the sphere");
    }
    if (max_analytical_solution == 0 || squared_analytical_l2_norm == 0) {
        throw std::runtime_error("The analytical solution has zero norm in the sphere interior");
    }

    const EMW::Types::scalar l2_error = std::sqrt(squared_l2_error);
    const EMW::Types::scalar analytical_l2_norm = std::sqrt(squared_analytical_l2_norm);
    return {
        .compared_cells = compared_cells,
        .c_norm = max_error,
        .relative_c_norm = max_error / max_analytical_solution,
        .l2_norm = l2_error,
        .relative_l2_norm = l2_error / analytical_l2_norm,
    };
}

} // namespace Research::VolDie::Mie
