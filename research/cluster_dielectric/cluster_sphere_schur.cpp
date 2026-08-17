#include "EMW/types/Types.hpp"

#include "mesh/volume_mesh/CubeMeshWithData.hpp"
#include "operators/volume/OperatorK.hpp"

#include "experiment/PhysicalCondition.hpp"
#include "math/integration/decart/Integration.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace EMW;

namespace {

constexpr Types::scalar SPHERE_EPSILON = 2.56;
constexpr Types::scalar SPHERE_RADIUS = 0.5;
constexpr Types::scalar CUBE_LENGTH = 2 * SPHERE_RADIUS;

constexpr Types::scalar RTOL = 1e-6;
constexpr Types::scalar ATOL = 1e-21;
constexpr Types::index LEVEL_2D = 2;
constexpr Types::index LEVEL_3D = 2;
constexpr Types::index LEVEL_4D = 2;
constexpr Types::index LEVEL_6D = 2;
constexpr Types::index NEARNESS_THRESHOLD = 2;

constexpr Types::scalar FREQUENCY_GHZ = 0.03;
constexpr Types::complex_d WAVE_NUMBER{Physics::get_k_on_frquency(FREQUENCY_GHZ), 0.0};

// This is the threshold used by VolumeOperatorMatrixReplacement in cluster_sphere.cpp.
constexpr Types::scalar MASK_THRESHOLD = 1e-7;

struct Options {
    Types::index nx = 16;
    std::string output_path;
    std::string singular_values_output_path;
    bool show_help = false;
};

std::string make_singular_values_output_path(const std::string &eigenvalues_output_path) {
    const std::filesystem::path eigenvalues_path{eigenvalues_output_path};
    const auto filename = eigenvalues_path.stem().string() + "_singular_values" + eigenvalues_path.extension().string();
    return (eigenvalues_path.parent_path() / filename).string();
}

Types::index parse_grid_size(std::string_view text) {
    std::size_t parsed_characters = 0;
    long long value = 0;
    try {
        value = std::stoll(std::string{text}, &parsed_characters);
    } catch (const std::exception &) {
        throw std::invalid_argument("Invalid Nx value: " + std::string{text});
    }

    if (parsed_characters != text.size() || value < 2 || value > std::numeric_limits<Types::index>::max()) {
        throw std::invalid_argument("Nx must be an integer greater than or equal to 2");
    }
    return static_cast<Types::index>(value);
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int arg_idx = 1; arg_idx < argc; ++arg_idx) {
        const std::string_view argument{argv[arg_idx]};
        if (argument == "--help" || argument == "-h") {
            options.show_help = true;
        } else if (argument == "--nx" || argument == "-n") {
            if (++arg_idx == argc)
                throw std::invalid_argument("Missing value after " + std::string{argument});
            options.nx = parse_grid_size(argv[arg_idx]);
        } else if (argument.starts_with("--nx=")) {
            options.nx = parse_grid_size(argument.substr(5));
        } else if (argument == "--output" || argument == "-o") {
            if (++arg_idx == argc)
                throw std::invalid_argument("Missing value after " + std::string{argument});
            options.output_path = argv[arg_idx];
        } else if (argument.starts_with("--output=")) {
            options.output_path = argument.substr(9);
        } else if (argument == "--singular-values-output" || argument == "--sv-output") {
            if (++arg_idx == argc)
                throw std::invalid_argument("Missing value after " + std::string{argument});
            options.singular_values_output_path = argv[arg_idx];
        } else if (argument.starts_with("--singular-values-output=")) {
            options.singular_values_output_path = argument.substr(25);
        } else if (argument.starts_with("--sv-output=")) {
            options.singular_values_output_path = argument.substr(12);
        } else {
            throw std::invalid_argument("Unknown command-line argument: " + std::string{argument});
        }
    }

    if (options.output_path.empty()) {
        options.output_path = "sphere_schur_eigenvalues_Nx" + std::to_string(options.nx) + ".csv";
    }
    if (options.singular_values_output_path.empty()) {
        options.singular_values_output_path = make_singular_values_output_path(options.output_path);
    }
    return options;
}

void print_help(const char *executable_name) {
    std::cout << "Usage: " << executable_name
              << " [--nx N] [--output FILE] [--singular-values-output FILE]\n\n"
              << "  -n, --nx N                    Set Nx = Ny = Nz (number of mesh nodes, default: 16).\n"
              << "                                The number of cells per axis is N - 1, as in cluster_sphere.cpp.\n"
              << "  -o, --output FILE             CSV file for eigenvalues.\n"
              << "      --sv-output FILE          CSV file for singular values.\n"
              << "      --singular-values-output  Same as --sv-output.\n"
              << "  -h, --help                    Show this help.\n";
}

struct CartesianCellIndex {
    Types::index x;
    Types::index y;
    Types::index z;
};

CartesianCellIndex to_cartesian_cell_index(Types::index linear_idx, Types::index cells_x,
                                            Types::index cells_y) {
    const Types::index x = linear_idx % cells_x;
    linear_idx /= cells_x;
    const Types::index y = linear_idx % cells_y;
    const Types::index z = linear_idx / cells_y;
    return {x, y, z};
}

template <typename ToeplitzMatrix>
Types::MatrixXc assemble_active_system_matrix(const ToeplitzMatrix &operator_k_matrix,
                                              const std::vector<Types::index> &active_cells,
                                              const std::vector<Types::complex_d> &contrast,
                                              Types::index cells_x, Types::index cells_y) {
    const Eigen::Index matrix_size = static_cast<Eigen::Index>(3 * active_cells.size());
    Types::MatrixXc system_matrix(matrix_size, matrix_size);

#pragma omp parallel for schedule(static)
    for (Eigen::Index active_row = 0; active_row < static_cast<Eigen::Index>(active_cells.size()); ++active_row) {
        const auto row_cell = active_cells[active_row];
        const auto row_idx = to_cartesian_cell_index(row_cell, cells_x, cells_y);

        for (Eigen::Index active_col = 0; active_col < static_cast<Eigen::Index>(active_cells.size()); ++active_col) {
            const auto col_cell = active_cells[active_col];
            const auto col_idx = to_cartesian_cell_index(col_cell, cells_x, cells_y);
            const auto &operator_block = operator_k_matrix.get_block(row_idx.z, col_idx.z)
                                             .get_block(row_idx.y, col_idx.y)
                                             .get_block(row_idx.x, col_idx.x);

            system_matrix.template block<3, 3>(3 * active_row, 3 * active_col) =
                -operator_block * contrast[col_cell];
        }

        system_matrix.template block<3, 3>(3 * active_row, 3 * active_row).diagonal().array() += 1.0;
    }

    return system_matrix;
}

void save_eigenvalues(const Types::VectorXc &eigenvalues, const std::string &output_path) {
    std::ofstream output{output_path};
    if (!output)
        throw std::runtime_error("Cannot open output file: " + output_path);

    output << "index,real,imag,abs\n";
    output << std::setprecision(std::numeric_limits<Types::scalar>::max_digits10);
    for (Eigen::Index idx = 0; idx < eigenvalues.size(); ++idx) {
        const auto value = eigenvalues[idx];
        output << idx << ',' << value.real() << ',' << value.imag() << ',' << std::abs(value) << '\n';
    }

    if (!output)
        throw std::runtime_error("Failed while writing output file: " + output_path);
}

void save_singular_values(const Types::VectorXd &singular_values, const std::string &output_path) {
    std::ofstream output{output_path};
    if (!output)
        throw std::runtime_error("Cannot open output file: " + output_path);

    output << "index,singular_value\n";
    output << std::setprecision(std::numeric_limits<Types::scalar>::max_digits10);
    for (Eigen::Index idx = 0; idx < singular_values.size(); ++idx) {
        output << idx << ',' << singular_values[idx] << '\n';
    }

    if (!output)
        throw std::runtime_error("Failed while writing output file: " + output_path);
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.show_help) {
            print_help(argv[0]);
            return 0;
        }

        Eigen::setNbThreads(1);

        const Types::index Nx = options.nx;
        const Types::index Ny = Nx;
        const Types::index Nz = Nx;

        Mesh::VolumeMesh::CubeMeshWithData mesh{
            Types::point_t{-CUBE_LENGTH / 2, -CUBE_LENGTH / 2, -CUBE_LENGTH / 2},
            CUBE_LENGTH,
            CUBE_LENGTH,
            CUBE_LENGTH,
            static_cast<std::size_t>(Nx),
            static_cast<std::size_t>(Ny),
            static_cast<std::size_t>(Nz)};
        mesh.setName("sphere_schur_" + std::to_string(Nx));

        const auto homogeneous_sphere = [](const Types::point_t &point) {
            return point.norm() < SPHERE_RADIUS ? Types::complex_d{-SPHERE_EPSILON, 0.8} : Types::complex_d{1.0, 0};
        };
        mesh.smoothScalarData<DecartIntegration::NewtonCotess::Quadrature<4, 4, 4>>("eps", homogeneous_sphere);

        const Types::scalar cube_measure = mesh.dx() * mesh.dy() * mesh.dz();
        const Types::scalar basis_fn_module = 1.0 / std::sqrt(cube_measure);

        std::cout << "Grid: Nx = Ny = Nz = " << Nx << " nodes (" << Nx - 1 << " cells per axis)\n"
                  << "Free-space wavelength = " << 2 * M_PI / WAVE_NUMBER.real() << '\n';

        Operators::Volume::operator_K_over_cube_mesh operator_k{WAVE_NUMBER, mesh};
        operator_k.set_tolerances(RTOL, ATOL);
        operator_k.set_adaptive_integration_max_levels({LEVEL_2D, LEVEL_3D, LEVEL_4D, LEVEL_6D});
        operator_k.set_nearness_threshold(NEARNESS_THRESHOLD);

        std::cout << "Assembling the Galerkin matrix of K..." << std::endl;
        const auto operator_k_matrix = operator_k.compute_galerkin_matrix(basis_fn_module);

        const auto &epsilon = mesh.getScalarData("eps");
        std::vector<Types::complex_d> contrast(epsilon.size());
        std::vector<Types::index> active_cells;
        active_cells.reserve(epsilon.size());
        for (Types::index cell_idx = 0; cell_idx < static_cast<Types::index>(epsilon.size()); ++cell_idx) {
            contrast[cell_idx] = epsilon[cell_idx] - 1.0;
            if (std::abs(contrast[cell_idx]) > MASK_THRESHOLD)
                active_cells.push_back(cell_idx);
        }

        if (active_cells.empty())
            throw std::runtime_error("The active part of the sphere is empty");

        const Eigen::Index active_unknowns = static_cast<Eigen::Index>(3 * active_cells.size());
        const long double bytes_per_dense_matrix = static_cast<long double>(active_unknowns) * active_unknowns *
                                                   sizeof(Types::complex_d);
        constexpr long double GIB = 1024.0L * 1024.0L * 1024.0L;
        std::cout << "Active cells: " << active_cells.size() << " of " << mesh.getCells().size() << '\n'
                  << "Dense system matrix size: " << active_unknowns << " x " << active_unknowns << '\n'
                  << "Memory per dense complex matrix: " << std::fixed << std::setprecision(2)
                  << bytes_per_dense_matrix / GIB << " GiB\n"
                  << "ComplexSchur needs several such matrices and O(n^3) operations." << std::endl;

        Types::VectorXc eigenvalues;
        Types::VectorXd singular_values;
        {
            std::cout << "Assembling A_active = I - K_active * diag(epsilon - 1)..." << std::endl;
            Types::MatrixXc system_matrix = assemble_active_system_matrix(operator_k_matrix, active_cells, contrast,
                                                                          Nx - 1, Ny - 1);

            std::cout << "Computing singular values (singular vectors are disabled)..." << std::endl;
            Eigen::BDCSVD<Types::MatrixXc> svd;
            svd.compute(system_matrix);
            if (svd.info() != Eigen::Success)
                throw std::runtime_error("The singular value decomposition did not converge");
            singular_values = svd.singularValues();

            std::cout << "Computing the complex Schur decomposition (Schur vectors are disabled)..." << std::endl;
            Eigen::ComplexSchur<Types::MatrixXc> schur_decomposition;
            schur_decomposition.compute(system_matrix, false);
            if (schur_decomposition.info() != Eigen::Success)
                throw std::runtime_error("The complex Schur decomposition did not converge");

            eigenvalues = schur_decomposition.matrixT().diagonal();
        }

        if (singular_values.size() == 0)
            throw std::runtime_error("The singular value decomposition returned no singular values");

        const Types::scalar sigma_max = singular_values[0];
        const Types::scalar sigma_min = singular_values[singular_values.size() - 1];
        const Types::scalar condition_number =
            sigma_min > 0.0 ? sigma_max / sigma_min : std::numeric_limits<Types::scalar>::infinity();

        std::cout << std::scientific << std::setprecision(std::numeric_limits<Types::scalar>::max_digits10)
                  << "Largest singular value: " << sigma_max << '\n'
                  << "Smallest singular value: " << sigma_min << '\n'
                  << "2-norm condition number: " << condition_number << std::endl;

        save_eigenvalues(eigenvalues, options.output_path);
        std::cout << "Saved " << eigenvalues.size() << " eigenvalues to " << options.output_path << std::endl;
        save_singular_values(singular_values, options.singular_values_output_path);
        std::cout << "Saved " << singular_values.size() << " singular values to "
                  << options.singular_values_output_path << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
