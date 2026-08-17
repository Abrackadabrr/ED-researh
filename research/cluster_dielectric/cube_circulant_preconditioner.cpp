#include "EMW/types/Types.hpp"

#include "experiment/PhysicalCondition.hpp"
#include "mesh/volume_mesh/CubeMesh.hpp"
#include "operators/volume/OperatorK.hpp"

#include "TripleBlockCirculant.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace EMW;

namespace {

constexpr Types::scalar DEFAULT_CUBE_LENGTH = 1.0;
constexpr Types::scalar DEFAULT_EPSILON = 2.56;

// lambda_0 = pi metres, hence k_0 = 2.
constexpr Types::scalar DEFAULT_FREQUENCY_GHZ = 0.5;

// OperatorK integration parameters are the same as in the current sphere case.
constexpr Types::scalar RTOL = 1e-6;
constexpr Types::scalar ATOL = 1e-21;
constexpr Types::index LEVEL_2D = 4;
constexpr Types::index LEVEL_3D = 4;
constexpr Types::index LEVEL_4D = 4;
constexpr Types::index LEVEL_6D = 4;
constexpr Types::index NEARNESS_THRESHOLD = 3;

constexpr Types::index BLOCK_SIZE = 3;
constexpr Types::index DEFAULT_NX = 16; // mesh nodes; there are Nx - 1 cells per axis
constexpr Types::index SAFE_EXACT_SPECTRUM_SIZE = 14000;

struct Options {
    Types::index nx = DEFAULT_NX;
    Types::scalar frequency_ghz = DEFAULT_FREQUENCY_GHZ;
    Types::scalar epsilon = DEFAULT_EPSILON;
    Types::scalar cube_length = DEFAULT_CUBE_LENGTH;
    std::string output_path;
    bool force_large_spectrum = false;
    bool show_help = false;
};

Types::index parse_index(std::string_view value, std::string_view option_name) {
    std::size_t parsed_characters = 0;
    unsigned long long parsed_value = 0;
    try {
        parsed_value = std::stoull(std::string{value}, &parsed_characters);
    } catch (const std::exception &) {
        throw std::invalid_argument("Invalid value for " + std::string{option_name} + ": " + std::string{value});
    }

    if (parsed_characters != value.size() || parsed_value > std::numeric_limits<Types::index>::max()) {
        throw std::invalid_argument("Invalid value for " + std::string{option_name} + ": " + std::string{value});
    }
    return static_cast<Types::index>(parsed_value);
}

Types::scalar parse_positive_scalar(std::string_view value, std::string_view option_name) {
    std::size_t parsed_characters = 0;
    Types::scalar parsed_value = 0;
    try {
        parsed_value = std::stod(std::string{value}, &parsed_characters);
    } catch (const std::exception &) {
        throw std::invalid_argument("Invalid value for " + std::string{option_name} + ": " + std::string{value});
    }

    if (parsed_characters != value.size() || !std::isfinite(parsed_value) || parsed_value <= 0) {
        throw std::invalid_argument(std::string{option_name} + " must be a positive finite number");
    }
    return parsed_value;
}

std::string_view require_value(int argc, char **argv, int &argument_index, std::string_view option_name) {
    if (++argument_index == argc)
        throw std::invalid_argument("Missing value after " + std::string{option_name});
    return argv[argument_index];
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int argument_index = 1; argument_index < argc; ++argument_index) {
        const std::string_view argument{argv[argument_index]};
        if (argument == "--help" || argument == "-h") {
            options.show_help = true;
        } else if (argument == "--force-large-spectrum") {
            options.force_large_spectrum = true;
        } else if (argument == "--nx" || argument == "-n") {
            options.nx = parse_index(require_value(argc, argv, argument_index, argument), argument);
        } else if (argument.starts_with("--nx=")) {
            options.nx = parse_index(argument.substr(5), "--nx");
        } else if (argument == "--frequency-ghz") {
            options.frequency_ghz =
                parse_positive_scalar(require_value(argc, argv, argument_index, argument), argument);
        } else if (argument.starts_with("--frequency-ghz=")) {
            options.frequency_ghz = parse_positive_scalar(argument.substr(16), "--frequency-ghz");
        } else if (argument == "--epsilon") {
            options.epsilon = parse_positive_scalar(require_value(argc, argv, argument_index, argument), argument);
        } else if (argument.starts_with("--epsilon=")) {
            options.epsilon = parse_positive_scalar(argument.substr(10), "--epsilon");
        } else if (argument == "--cube-length") {
            options.cube_length =
                parse_positive_scalar(require_value(argc, argv, argument_index, argument), argument);
        } else if (argument.starts_with("--cube-length=")) {
            options.cube_length = parse_positive_scalar(argument.substr(14), "--cube-length");
        } else if (argument == "--output" || argument == "-o") {
            options.output_path = require_value(argc, argv, argument_index, argument);
        } else if (argument.starts_with("--output=")) {
            options.output_path = argument.substr(9);
        } else {
            throw std::invalid_argument("Unknown command-line argument: " + std::string{argument});
        }
    }

    // Eigen's default FFT backend does not support transforms of length one.
    if (options.nx < 3)
        throw std::invalid_argument("Nx must be at least 3 (at least two cells per axis)");
    if (options.epsilon <= 1.0)
        throw std::invalid_argument("Epsilon must be greater than one for this dielectric-cube example");
    if (options.output_path.empty()) {
        options.output_path =
            "cube_chan_preconditioned_eigenvalues_Nx" + std::to_string(options.nx) + ".csv";
    }
    return options;
}

void print_help(const char *executable_name) {
    std::cout << "Usage: " << executable_name << " [options]\n\n"
              << "  -n, --nx N                 Nx = Ny = Nz mesh nodes, N >= 3 (default: " << DEFAULT_NX << ")\n"
              << "  --frequency-ghz F          frequency in GHz (default gives lambda_0 = pi)\n"
              << "  --epsilon E                homogeneous cube permittivity (default: " << DEFAULT_EPSILON << ")\n"
              << "  --cube-length L             cube side length in metres (default: " << DEFAULT_CUBE_LENGTH << ")\n"
              << "  -o, --output FILE           CSV file for eigenvalues\n"
              << "  --force-large-spectrum      allow an exact dense spectrum above "
              << SAFE_EXACT_SPECTRUM_SIZE << " unknowns\n"
              << "  -h, --help                  show this help\n";
}

struct CellIndex3D {
    Types::index x;
    Types::index y;
    Types::index z;
};

CellIndex3D unflatten(Types::index index, Types::index nx, Types::index ny) noexcept {
    const Types::index x = index % nx;
    index /= nx;
    const Types::index y = index % ny;
    const Types::index z = index / ny;
    return {x, y, z};
}

template <typename TripleToeplitzMatrix>
Types::MatrixXc assemble_dense_system_matrix(const TripleToeplitzMatrix &operator_k, Types::index nx,
                                             Types::index ny, Types::index nz, Types::complex_d contrast) {
    const Types::index cells = nx * ny * nz;
    const Eigen::Index matrix_size = static_cast<Eigen::Index>(BLOCK_SIZE * cells);
    Types::MatrixXc system_matrix(matrix_size, matrix_size);

#pragma omp parallel for schedule(static)
    for (Types::index row_cell = 0; row_cell < cells; ++row_cell) {
        const auto row = unflatten(row_cell, nx, ny);
        for (Types::index col_cell = 0; col_cell < cells; ++col_cell) {
            const auto col = unflatten(col_cell, nx, ny);
            system_matrix.block<3, 3>(3 * row_cell, 3 * col_cell) =
                -contrast * operator_k.get_block(row.z, col.z, row.y, col.y, row.x, col.x);
        }
        system_matrix.block<3, 3>(3 * row_cell, 3 * row_cell).diagonal().array() += 1.0;
    }
    return system_matrix;
}

Types::VectorXc make_test_vector(Types::index size) {
    Types::VectorXc vector(size);
    for (Types::index index = 0; index < size; ++index) {
        const Types::scalar argument = static_cast<Types::scalar>(index + 1);
        vector(index) = Types::complex_d{std::sin(0.17 * argument) + 0.1 * std::cos(0.031 * argument),
                                         std::cos(0.11 * argument) - 0.2 * std::sin(0.047 * argument)};
    }
    return vector;
}

void save_eigenvalues(const Types::VectorXc &eigenvalues, const std::string &output_path) {
    std::ofstream output{output_path};
    if (!output)
        throw std::runtime_error("Cannot open output file: " + output_path);

    output << "index,real,imag,abs,distance_to_one\n";
    output << std::setprecision(std::numeric_limits<Types::scalar>::max_digits10);
    for (Eigen::Index index = 0; index < eigenvalues.size(); ++index) {
        const Types::complex_d eigenvalue = eigenvalues(index);
        output << index << ',' << eigenvalue.real() << ',' << eigenvalue.imag() << ',' << std::abs(eigenvalue) << ','
               << std::abs(eigenvalue - Types::complex_d{1.0, 0.0}) << '\n';
    }
    if (!output)
        throw std::runtime_error("Failed while writing eigenvalues to: " + output_path);
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

        const Types::index cells_x = options.nx - 1;
        const Types::index cells_y = cells_x;
        const Types::index cells_z = cells_x;
        const Types::index total_cells = cells_x * cells_y * cells_z;
        const Types::index unknowns = BLOCK_SIZE * total_cells;

        if (unknowns > SAFE_EXACT_SPECTRUM_SIZE && !options.force_large_spectrum) {
            throw std::runtime_error("The exact spectrum would require a dense " + std::to_string(unknowns) +
                                     " x " + std::to_string(unknowns) +
                                     " matrix. Pass --force-large-spectrum if this is intentional");
        }

        const long double bytes_per_matrix =
            static_cast<long double>(unknowns) * unknowns * sizeof(Types::complex_d);
        constexpr long double GIB = 1024.0L * 1024.0L * 1024.0L;

        const Types::complex_d wave_number{Physics::get_k_on_frquency(options.frequency_ghz), 0.0};
        const Types::complex_d contrast{options.epsilon - 1.0, 0.0};

        std::cout << std::setprecision(10)
                  << "Grid: Nx = Ny = Nz = " << options.nx << " nodes (" << cells_x << " cells per axis)\n"
                  << "Unknowns: " << unknowns << '\n'
                  << "Frequency: " << options.frequency_ghz << " GHz\n"
                  << "Free-space wavelength: "
                  << 2.0 * Math::Constants::PI<Types::scalar>() / wave_number.real() << " m\n"
                  << "Cube epsilon: " << options.epsilon << '\n'
                  << "Memory per dense complex matrix: " << std::fixed << std::setprecision(3)
                  << bytes_per_matrix / GIB << " GiB\n"
                  << std::defaultfloat << std::setprecision(10);

        Mesh::VolumeMesh::CubeMesh mesh{
            Types::point_t{-options.cube_length / 2.0, -options.cube_length / 2.0,
                           -options.cube_length / 2.0},
            options.cube_length,
            options.cube_length,
            options.cube_length,
            options.nx,
            options.nx,
            options.nx};

        const Types::scalar cell_measure = mesh.dx() * mesh.dy() * mesh.dz();
        const Types::scalar basis_function_module = 1.0 / std::sqrt(cell_measure);

        Operators::Volume::operator_K_over_cube_mesh operator_k{wave_number, mesh};
        operator_k.set_tolerances(RTOL, ATOL);
        operator_k.set_adaptive_integration_max_levels({LEVEL_2D, LEVEL_3D, LEVEL_4D, LEVEL_6D});
        operator_k.set_nearness_threshold(NEARNESS_THRESHOLD);

        std::cout << "Assembling the three-level block-Toeplitz Galerkin matrix K..." << std::endl;
        const auto operator_k_matrix = operator_k.compute_galerkin_matrix_new(basis_function_module);

        std::cout << "Building the optimal Chan block-circulant preconditioner C..." << std::endl;
        auto first_block_column =
            Research::VolDie::build_chan_circulant_first_block_column<Types::complex_d>(
                operator_k_matrix, cells_x, cells_y, cells_z, contrast);
        using Circulant = Research::VolDie::TripleBlockCirculant3x3FFT<Types::complex_d>;
        const Circulant preconditioner{cells_x, cells_y, cells_z, std::move(first_block_column)};

        const auto worst_frequency = preconditioner.worst_conditioned_frequency();
        std::cout << "Minimum reciprocal condition number of the 3x3 Fourier symbols: "
                  << preconditioner.minimum_reciprocal_condition() << " at (" << worst_frequency.x << ", "
                  << worst_frequency.y << ", " << worst_frequency.z << ")\n";

        const Types::VectorXc test_vector = make_test_vector(unknowns);
        const Types::VectorXc fft_product = preconditioner.multiply(test_vector);
        const Types::VectorXc direct_product = preconditioner.multiply_direct(test_vector);
        const Types::scalar multiplication_error =
            (fft_product - direct_product).norm() / std::max<Types::scalar>(direct_product.norm(), 1.0);
        const Types::scalar inverse_error =
            (preconditioner.solve(fft_product) - test_vector).norm() / test_vector.norm();
        std::cout << std::scientific << std::setprecision(3)
                  << "Relative error of FFT circulant multiplication: " << multiplication_error << '\n'
                  << "Relative error of C^{-1}(C*x): " << inverse_error << '\n'
                  << std::defaultfloat << std::setprecision(10);

        if (multiplication_error > 1e-10 || inverse_error > 1e-8)
            throw std::runtime_error("The FFT circulant validation failed");

        std::cout << "Assembling the dense full-cube system A = I - (epsilon - 1) K..." << std::endl;
        const Types::MatrixXc system_matrix =
            assemble_dense_system_matrix(operator_k_matrix, cells_x, cells_y, cells_z, contrast);

        std::cout << "Applying C^{-1} to every column of A through FFT..." << std::endl;
        Types::MatrixXc preconditioned_matrix(unknowns, unknowns);
        for (Types::index column = 0; column < unknowns; ++column)
            preconditioned_matrix.col(column) = preconditioner.solve(system_matrix.col(column));

        const Types::scalar preconditioned_action_error =
            (preconditioned_matrix * test_vector - preconditioner.solve(system_matrix * test_vector)).norm() /
            (preconditioned_matrix * test_vector).norm();
        std::cout << std::scientific << std::setprecision(3)
                  << "Relative consistency error for C^{-1}A: " << preconditioned_action_error << '\n'
                  << std::defaultfloat << std::setprecision(10);

        std::cout << "Computing the exact spectrum of C^{-1}A by complex Schur decomposition..." << std::endl;
        Eigen::ComplexSchur<Types::MatrixXc> schur_decomposition;
        schur_decomposition.compute(preconditioned_matrix, false);
        if (schur_decomposition.info() != Eigen::Success)
            throw std::runtime_error("The complex Schur decomposition did not converge");

        const Types::VectorXc eigenvalues = schur_decomposition.matrixT().diagonal();
        save_eigenvalues(eigenvalues, options.output_path);

        Types::scalar maximum_distance_to_one = 0.0;
        Types::scalar mean_distance_to_one = 0.0;
        for (const auto &eigenvalue : eigenvalues) {
            const Types::scalar distance = std::abs(eigenvalue - Types::complex_d{1.0, 0.0});
            maximum_distance_to_one = std::max(maximum_distance_to_one, distance);
            mean_distance_to_one += distance;
        }
        mean_distance_to_one /= static_cast<Types::scalar>(eigenvalues.size());

        std::cout << "Mean |lambda - 1|: " << mean_distance_to_one << '\n'
                  << "Max  |lambda - 1|: " << maximum_distance_to_one << '\n'
                  << "Saved " << eigenvalues.size() << " eigenvalues to " << options.output_path << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
