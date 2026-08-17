// Frequency sweep of the extinction cross section of a homogeneous dielectric sphere.

#include "EMW/types/Types.hpp"

#include "experiment/PhysicalCondition.hpp"
#include "math/fourier/TripleToeplitz3x3Fourier.hpp"
#include "math/integration/decart/Integration.hpp"
#include "mesh/volume_mesh/CubeMeshWithData.hpp"
#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "../Solve.hpp"
#include "MatrixReplacement.hpp"
#include "MatrixTraits.hpp"
#include "ClusterSphereExtinctionConfig.hpp"
#include "utility/YamlParser.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace EMW;

namespace {

using Configuration = Research::ClusterDielectric::ClusterSphereExtinctionConfig;

constexpr const char *CONFIGURATION_FILE_NAME = "cluster_sphere_extinction.yaml";

struct FrequencyResult {
    Types::scalar extinction_cross_section{};
    Types::scalar field_l2_norm{};
};

const char *environment_value(const char *name) {
    const char *value = std::getenv(name);
    return value == nullptr ? "<not set>" : value;
}

bool report_openmp_configuration(int mpi_rank, int mpi_process_count) {
#ifndef _OPENMP
    if (mpi_rank == 0) {
        std::cerr << "OpenMP is disabled for cluster_sphere_extinction.cpp: the compiler did not define "
                     "_OPENMP. Linking an OpenMP runtime is not sufficient; this translation unit must be "
                     "compiled with OpenMP enabled (for GCC/Clang, normally -fopenmp)."
                  << std::endl;
    }
    return false;
#else
    int actual_team_size = 0;
#pragma omp parallel reduction(+ : actual_team_size)
    { actual_team_size += 1; }

    // Print one complete line per rank. The barriers keep diagnostics from different
    // MPI processes readable and also expose rank-specific Slurm CPU bindings.
    for (int rank = 0; rank < mpi_process_count; ++rank) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == mpi_rank) {
            std::ostringstream message;
            message << "[MPI rank " << mpi_rank << "] OpenMP " << _OPENMP
                    << ": actual team = " << actual_team_size
                    << ", max threads = " << omp_get_max_threads()
                    << ", available processors = " << omp_get_num_procs()
                    << ", dynamic = " << omp_get_dynamic()
                    << ", thread limit = " << omp_get_thread_limit()
                    << "; OMP_NUM_THREADS=" << environment_value("OMP_NUM_THREADS")
                    << ", OMP_DYNAMIC=" << environment_value("OMP_DYNAMIC")
                    << ", OMP_PROC_BIND=" << environment_value("OMP_PROC_BIND")
                    << ", OMP_PLACES=" << environment_value("OMP_PLACES")
                    << ", SLURM_CPUS_PER_TASK=" << environment_value("SLURM_CPUS_PER_TASK")
                    << ", SLURM_CPU_BIND=" << environment_value("SLURM_CPU_BIND");
            std::cout << message.str() << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (actual_team_size == 1 && mpi_rank == 0) {
        std::cerr << "OpenMP is compiled in, but the runtime created only one thread. Check "
                     "OMP_NUM_THREADS, OMP_DYNAMIC and the Slurm --cpus-per-task/CPU-binding settings above."
                  << std::endl;
    }
    return true;
#endif
}

std::filesystem::path configuration_path(int argc, char **argv) {
    if (argc > 2) {
        throw std::invalid_argument("Usage: " + std::string{argv[0]} + " [configuration.yaml]");
    }
    if (argc == 2) {
        return argv[1];
    }

    const std::filesystem::path working_directory_candidate = CONFIGURATION_FILE_NAME;
    if (std::filesystem::exists(working_directory_candidate)) {
        return working_directory_candidate;
    }

    const std::filesystem::path executable_candidate =
        std::filesystem::absolute(argv[0]).parent_path() / CONFIGURATION_FILE_NAME;
    if (std::filesystem::exists(executable_candidate)) {
        return executable_candidate;
    }

#ifdef CLUSTER_SPHERE_EXTINCTION_SOURCE_CONFIG
    const std::filesystem::path source_candidate = CLUSTER_SPHERE_EXTINCTION_SOURCE_CONFIG;
    if (std::filesystem::exists(source_candidate)) {
        return source_candidate;
    }
#endif

    // Return the conventional path so the YAML loader produces one clear error.
    return working_directory_candidate;
}

Types::index cells_per_axis(Types::scalar wave_number, const Configuration &config) {
    // h <= lambda_inside / CELLS_PER_INTERNAL_WAVELENGTH,
    // lambda_inside = 2*pi/(k_0*sqrt(|epsilon_r|)).
    const Types::scalar cube_length = 2.0 * config.sphere.radius;
    const Types::scalar internal_wave_number = wave_number * std::sqrt(std::abs(config.sphere.epsilon));
    const Types::scalar wavelength_limited_cells =
        std::ceil(config.mesh.cells_per_internal_wavelength * cube_length * internal_wave_number /
                  (2.0 * M_PI));
    return std::max(config.mesh.min_cells_per_axis,
                    static_cast<Types::index>(wavelength_limited_cells));
}

Types::scalar calculate_extinction_cross_section(const Types::VectorXc &incident_projection,
                                                 const Types::VectorXc &solution,
                                                 const Types::VectorXc &epsilon_minus_one,
                                                 Types::scalar wave_number,
                                                 const Types::Vector3d &polarization) {
    // For a unit-amplitude incident wave, the volume form of the optical theorem is
    // C_ext = k_0 Im int_V conj(E_inc) . (epsilon_r - 1) E dV.
    // The basis is L2-normalized, so the integral is exactly the coefficient-space
    // scalar product below (up to the Galerkin discretization error).
    const Types::complex_d extinction_integral =
        incident_projection.dot(epsilon_minus_one.cwiseProduct(solution));
    return wave_number * std::imag(extinction_integral) / polarization.squaredNorm();
}

FrequencyResult calculate_at_frequency(Types::scalar frequency_ghz,
                                       int mpi_rank,
                                       const Configuration &config) {
    const auto homogeneous_sphere = [&config](const Types::point_t &point) {
        return point.norm() < config.sphere.radius ? config.sphere.epsilon
                                                    : Types::complex_d{1.0, 0.0};
    };

    const Types::complex_d wave_number{Physics::get_k_on_frquency(frequency_ghz), 0.0};
    const Types::scalar cube_length = 2.0 * config.sphere.radius;
    const Types::index cell_count = cells_per_axis(wave_number.real(), config);
    const Types::index Nx = cell_count + 1;
    const Types::index Ny = Nx;
    const Types::index Nz = Nx;

    Mesh::VolumeMesh::CubeMeshWithData mesh{
        Types::point_t{-cube_length / 2.0, -cube_length / 2.0, -cube_length / 2.0},
        cube_length,
        cube_length,
        cube_length,
        Nx,
        Ny,
        Nz};
    mesh.smoothScalarData<DecartIntegration::NewtonCotess::Quadrature<1, 1, 1>>("eps", homogeneous_sphere);

    const Types::scalar internal_wavelength =
        2.0 * M_PI / (wave_number.real() * std::sqrt(std::abs(config.sphere.epsilon)));
    const Types::scalar cells_per_internal_wavelength = internal_wavelength / mesh.h();
    std::cout << "\n[MPI rank " << mpi_rank << "] frequency = " << frequency_ghz << " GHz\n"
              << "[MPI rank " << mpi_rank << "] Nx = " << Nx << ", cells per axis = " << cell_count << '\n'
              << "[MPI rank " << mpi_rank
              << "] internal wavelength / mesh.h = " << cells_per_internal_wavelength << std::endl;

    // Galerkin projection of the incident field onto the piecewise-constant basis.
    Physics::planeWaveCase incident_field{
        config.incident_wave.polarization, wave_number, config.incident_wave.direction};
    const Types::scalar cell_measure = mesh.dx() * mesh.dy() * mesh.dz();
    const Types::scalar basis_function_module = 1.0 / std::sqrt(cell_measure);
    Operators::Volume::ProjectorOnMesh projector{mesh};
    const auto projected_rhs =
        projector([incident_field](Types::point_t point) { return incident_field.value(point); });
    Types::VectorXc rhs = projected_rhs * basis_function_module;

    // The same Galerkin volume operator and FFT Toeplitz multiplication as in cluster_sphere.cpp.
    Operators::Volume::operator_K_over_cube_mesh operator_k{wave_number, mesh};
    operator_k.set_tolerances(config.integration.relative_tolerance,
                              config.integration.absolute_tolerance);
    operator_k.set_adaptive_integration_max_levels({config.integration.level_2d,
                                                    config.integration.level_3d,
                                                    config.integration.level_4d,
                                                    config.integration.level_6d});
    operator_k.set_nearness_threshold(config.integration.nearness_threshold);
    const auto operator_k_matrix = operator_k.compute_galerkin_matrix(basis_function_module);
    std::cout << "[MPI rank " << mpi_rank << "] matrix sizes: " << operator_k_matrix.rows() << " x "
              << operator_k_matrix.cols() << std::endl;

    using FourierOperator = Math::Fourier::TripleToeplitz3x3FourierParallel<Types::complex_d>;
    const FourierOperator fourier_operator{operator_k_matrix};

    Types::VectorXc epsilon_minus_one = Types::VectorXc::Zero(3 * mesh.getCells().size());
    const auto epsilon_data = mesh.getScalarData("eps");
    for (std::size_t cell = 0; cell < mesh.getCells().size(); ++cell) {
        const Types::complex_d contrast = epsilon_data[cell] - Types::complex_d{1.0, 0.0};
        epsilon_minus_one.segment<3>(3 * cell).setConstant(contrast);
    }

    Math::LinAgl::Matrix::Wrappers::VolumeOperatorMatrixReplacement system{fourier_operator,
                                                                            epsilon_minus_one};
    rhs = system.modify_rhs_according_to_mask(rhs);
    const Types::VectorXc solution = Research::solve<Eigen::GMRES>(
        system,
        rhs,
        config.gmres.max_iterations,
        config.gmres.tolerance,
        config.gmres.restart);

    const Types::scalar extinction_cross_section = calculate_extinction_cross_section(
        rhs,
        solution,
        system.get_epsilon_vec(),
        wave_number.real(),
        config.incident_wave.polarization);

    // The piecewise-constant basis is L2-normalized and the masked solution is zero
    // outside the sphere, hence the Euclidean coefficient norm is the field L2 norm.
    const Types::scalar field_l2_norm = solution.norm();

    std::cout << "[MPI rank " << mpi_rank << "] extinction cross section = "
              << extinction_cross_section << " m^2\n"
              << "[MPI rank " << mpi_rank << "] field L2 norm = " << field_l2_norm << std::endl;

    return {.extinction_cross_section = extinction_cross_section, .field_l2_norm = field_l2_norm};
}

} // namespace

int main(int argc, char **argv) {
    static_assert(std::is_same_v<Types::scalar, double>, "MPI_DOUBLE must match Types::scalar");

    int provided_thread_level = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level) != MPI_SUCCESS) {
        std::cerr << "MPI initialization failed" << std::endl;
        return 1;
    }

    int mpi_rank = 0;
    int mpi_process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_process_count);

    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (mpi_rank == 0) {
            std::cerr << "MPI implementation does not provide MPI_THREAD_FUNNELED" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (!report_openmp_configuration(mpi_rank, mpi_process_count)) {
        MPI_Finalize();
        return 1;
    }

    Eigen::setNbThreads(1);

    try {
        const std::filesystem::path config_path = configuration_path(argc, argv);
        const YAML::Node yaml = Utility::Yaml::parse_file(config_path);
        const Configuration config =
            Research::ClusterDielectric::cluster_sphere_extinction_config_from_yaml(yaml);
        if (mpi_rank == 0) {
            std::cout << "Configuration loaded from " << config_path << std::endl;
        }

        const Types::index frequency_count = static_cast<Types::index>(
            std::floor((config.frequency_sweep.end_ghz - config.frequency_sweep.start_ghz) /
                           config.frequency_sweep.step_ghz +
                       0.5)) +
            1;
        if (frequency_count > static_cast<Types::index>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("The frequency count exceeds MPI_Gatherv limits");
        }

        // Static cyclic (round-robin) distribution: rank r evaluates r, r+P, r+2P, ... .
        // Thus expensive high-frequency iterations are interleaved rather than assigned
        // as one contiguous batch to a single MPI process.
        std::vector<Types::index> local_frequency_indices;
        for (Types::index frequency_index = static_cast<Types::index>(mpi_rank);
             frequency_index < frequency_count;
             frequency_index += static_cast<Types::index>(mpi_process_count)) {
            local_frequency_indices.push_back(frequency_index);
        }

        std::vector<Types::scalar> local_extinction_cross_sections;
        std::vector<Types::scalar> local_field_l2_norms;
        local_extinction_cross_sections.reserve(local_frequency_indices.size());
        local_field_l2_norms.reserve(local_frequency_indices.size());

        for (const Types::index frequency_index : local_frequency_indices) {
            const Types::scalar frequency_ghz =
                config.frequency_sweep.start_ghz +
                frequency_index * config.frequency_sweep.step_ghz;
            const FrequencyResult result = calculate_at_frequency(frequency_ghz, mpi_rank, config);
            local_extinction_cross_sections.push_back(result.extinction_cross_section);
            local_field_l2_norms.push_back(result.field_l2_norm);
        }

        const int local_result_count = static_cast<int>(local_extinction_cross_sections.size());
        std::vector<int> receive_counts(mpi_rank == 0 ? mpi_process_count : 0);
        MPI_Gather(&local_result_count,
                   1,
                   MPI_INT,
                   receive_counts.data(),
                   1,
                   MPI_INT,
                   0,
                   MPI_COMM_WORLD);

        std::vector<int> displacements(mpi_rank == 0 ? mpi_process_count : 0);
        std::vector<Types::scalar> gathered_extinction_cross_sections;
        std::vector<Types::scalar> gathered_field_l2_norms;
        if (mpi_rank == 0) {
            int gathered_count = 0;
            for (int process = 0; process < mpi_process_count; ++process) {
                displacements[process] = gathered_count;
                gathered_count += receive_counts[process];
            }
            gathered_extinction_cross_sections.resize(gathered_count);
            gathered_field_l2_norms.resize(gathered_count);
        }

        MPI_Gatherv(local_extinction_cross_sections.data(),
                    local_result_count,
                    MPI_DOUBLE,
                    gathered_extinction_cross_sections.data(),
                    receive_counts.data(),
                    displacements.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(local_field_l2_norms.data(),
                    local_result_count,
                    MPI_DOUBLE,
                    gathered_field_l2_norms.data(),
                    receive_counts.data(),
                    displacements.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            std::vector<Types::scalar> extinction_cross_sections(frequency_count);
            std::vector<Types::scalar> field_l2_norms(frequency_count);

            // Undo the cyclic per-rank layout produced by MPI_Gatherv.
            for (int process = 0; process < mpi_process_count; ++process) {
                for (int local_index = 0; local_index < receive_counts[process]; ++local_index) {
                    const Types::index frequency_index =
                        static_cast<Types::index>(process) +
                        static_cast<Types::index>(local_index) * static_cast<Types::index>(mpi_process_count);
                    const int gathered_index = displacements[process] + local_index;
                    extinction_cross_sections[frequency_index] =
                        gathered_extinction_cross_sections[gathered_index];
                    field_l2_norms[frequency_index] = gathered_field_l2_norms[gathered_index];
                }
            }

            const auto write_results = [&](std::ostream &stream) {
                stream << "frequency_GHz,extinction_cross_section_m2,field_l2_norm,Nx,"
                          "cells_per_internal_wavelength\n";
                stream << std::setprecision(16);
                for (Types::index frequency_index = 0; frequency_index < frequency_count; ++frequency_index) {
                    const Types::scalar frequency_ghz =
                        config.frequency_sweep.start_ghz +
                        frequency_index * config.frequency_sweep.step_ghz;
                    const Types::scalar wave_number = Physics::get_k_on_frquency(frequency_ghz);
                    const Types::index cell_count = cells_per_axis(wave_number, config);
                    const Types::index Nx = cell_count + 1;
                    const Types::scalar internal_wavelength =
                        2.0 * M_PI /
                        (wave_number * std::sqrt(std::abs(config.sphere.epsilon)));
                    const Types::scalar cube_length = 2.0 * config.sphere.radius;
                    const Types::scalar cells_per_internal_wavelength =
                        internal_wavelength /
                        (cube_length / static_cast<Types::scalar>(cell_count));
                    stream << frequency_ghz << ',' << extinction_cross_sections[frequency_index] << ','
                           << field_l2_norms[frequency_index] << ',' << Nx << ','
                           << cells_per_internal_wavelength << '\n';
                }
            };

            std::ofstream output{config.output_file};
            if (output) {
                write_results(output);
                output.flush();
            }

            if (output) {
                std::cout << "\nResults saved to " << config.output_file << std::endl;
            } else {
                std::cerr << "\nCannot write results to " << config.output_file
                          << ". Printing them to stdout instead.\n";
                write_results(std::cout);
                std::cout.flush();
            }
        }

        MPI_Finalize();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[MPI rank " << mpi_rank << "] " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
