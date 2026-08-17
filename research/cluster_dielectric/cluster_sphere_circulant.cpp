#include "EMW/types/Types.hpp"

#include "experiment/ESA.hpp"
#include "experiment/PhysicalCondition.hpp"
#include "math/fields/Utils.hpp"
#include "math/fourier/TripleToeplitz3x3Fourier.hpp"
#include "math/integration/decart/Integration.hpp"
#include "mesh/Utils.hpp"
#include "mesh/volume_mesh/CubeMeshWithData.hpp"
#include "operators/volume/OperatorK.hpp"
#include "operators/volume/ProjectorOnMesh.hpp"

#include "../Solve.hpp"
#include "MatrixReplacement.hpp"
#include "MatrixTraits.hpp"
#include "TripleBlockCirculant.hpp"
#include "Utils.hpp"
#include "analytical_solution/SphereMieAnalytical.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

using namespace EMW;

namespace {

constexpr Types::scalar SPHERE_EPSILON = 4;
constexpr Types::scalar SPHERE_RADIUS = 0.5;
constexpr Types::scalar CUBE_LENGTH = 2 * SPHERE_RADIUS;
constexpr Types::index NX = 61;

constexpr Types::scalar RTOL = 1e-6;
constexpr Types::scalar ATOL = 1e-21;
constexpr Types::index LEVEL_2D = 2;
constexpr Types::index LEVEL_3D = 2;
constexpr Types::index LEVEL_4D = 2;
constexpr Types::index LEVEL_6D = 2;
constexpr Types::index NEARNESS_THRESHOLD = 3;

constexpr Types::scalar FREQUENCY_GHZ = 1.0;
constexpr Types::complex_d WAVE_NUMBER{Physics::get_k_on_frquency(FREQUENCY_GHZ), 0.0};
constexpr Types::Vector3d POLARIZATION{1.0, 0.0, 0.0};
constexpr Types::Vector3d WAVE_VECTOR{0.0, 0.0, -1.0};

constexpr Types::index GMRES_MAX_ITERATIONS = 10000;
constexpr Types::scalar GMRES_TOLERANCE = 1e-9;
constexpr Types::index GMRES_RESTART = 4e9 / (NX * NX * NX * 16);

const std::string OUTPUT_DIRECTORY =
    "/home/evgen/Education/MasterDegree/thesis/ED-researh/research/cluster_dielectric/";

} // namespace

int main() {
    Eigen::setNbThreads(1);

    const auto homogeneous_sphere = [](const Types::point_t &point) {
        return point.norm() < SPHERE_RADIUS ? SPHERE_EPSILON : 1.0;
    };

    const auto homogeneous_cube = [](const Types::point_t &point) {
        return SPHERE_EPSILON;
    };

    constexpr Types::index NY = NX;
    constexpr Types::index NZ = NX;
    constexpr Types::index CELLS_PER_AXIS = NX - 1;

    // 1. Build the enclosing cube and smooth the discontinuous sphere contrast on its cells.
    Mesh::VolumeMesh::CubeMeshWithData mesh{Types::point_t{-CUBE_LENGTH / 2.0, -CUBE_LENGTH / 2.0, -CUBE_LENGTH / 2.0},
                                            CUBE_LENGTH,
                                            CUBE_LENGTH,
                                            CUBE_LENGTH,
                                            NX,
                                            NY,
                                            NZ};
    mesh.setName("sphere_" + std::to_string(NX));
    mesh.smoothScalarData<DecartIntegration::NewtonCotess::Quadrature<1, 1, 1>>("eps", homogeneous_sphere);

    std::cout << CELLS_PER_AXIS << " cubes per size\n"
              << "Free-space wavelength = " << 2.0 * M_PI / WAVE_NUMBER.real() << '\n'
              << "lambda_0 / mesh.h = " << 2.0 * M_PI / (WAVE_NUMBER.real() * CUBE_LENGTH / CELLS_PER_AXIS) << '\n'
              << "lambda / mesh.h = "
              << 2.0 * M_PI / (std::sqrt(SPHERE_EPSILON) * WAVE_NUMBER.real() * CUBE_LENGTH / CELLS_PER_AXIS)
              << std::endl;

    // 2. Project the x-polarized incident plane wave onto piecewise-constant basis functions.
    Physics::planeWaveCase incident_field{POLARIZATION, WAVE_NUMBER, WAVE_VECTOR};
    const Types::scalar cell_measure = mesh.dx() * mesh.dy() * mesh.dz();
    const Types::scalar basis_function_module = 1.0 / std::sqrt(cell_measure);
    Operators::Volume::ProjectorOnMesh projector{mesh};
    const auto projected_rhs =
        projector([incident_field](Types::point_t point) { return incident_field.value(point); });
    Types::VectorXc rhs = projected_rhs * basis_function_module;

    // 3. Assemble the three-level block-Toeplitz Galerkin matrix K.
    Operators::Volume::operator_K_over_cube_mesh operator_k{WAVE_NUMBER, mesh};
    operator_k.set_tolerances(RTOL, ATOL);
    operator_k.set_adaptive_integration_max_levels({LEVEL_2D, LEVEL_3D, LEVEL_4D, LEVEL_6D});
    operator_k.set_nearness_threshold(NEARNESS_THRESHOLD);
    const auto operator_k_matrix = operator_k.compute_galerkin_matrix(basis_function_module);
    std::cout << "Matrix sizes: " << operator_k_matrix.rows() << " x " << operator_k_matrix.cols() << std::endl;

    // The Toeplitz K action is evaluated by a zero-padded FFT, as in the original sphere task.
    using FourierOperator = Math::Fourier::TripleToeplitz3x3FourierParallel<Types::complex_d>;
    const FourierOperator fourier_operator{operator_k_matrix};

    // 4. Store D = diag(epsilon - 1); the matrix wrapper derives the active-cell mask from it.
    Types::VectorXc epsilon_minus_one = Types::VectorXc::Zero(3 * mesh.getCells().size());
    const auto epsilon_data = mesh.getScalarData("eps");
    for (std::size_t cell = 0; cell < mesh.getCells().size(); ++cell) {
        const Types::complex_d contrast = epsilon_data[cell] - Types::complex_d{1.0, 0.0};
        epsilon_minus_one.segment<3>(3 * cell).setConstant(contrast);
    }

    // 5. Form the full-cube Chan approximation C of I-(epsilon_sphere-1)K.
    // Every cell of the enclosing cube participates in C; the sphere mask is deliberately not used.
    auto first_block_column = Research::VolDie::build_chan_circulant_first_block_column<Types::complex_d>(
        operator_k_matrix, CELLS_PER_AXIS, CELLS_PER_AXIS, CELLS_PER_AXIS, Types::complex_d{SPHERE_EPSILON - 1.0, 0.0});
    using Circulant = Research::VolDie::TripleBlockCirculant3x3FFT<Types::complex_d>;
    const Circulant circulant{CELLS_PER_AXIS, CELLS_PER_AXIS, CELLS_PER_AXIS, std::move(first_block_column)};

    const auto worst_frequency = circulant.worst_conditioned_frequency();
    std::cout << std::setprecision(10)
              << "Minimum rcond of the circulant 3x3 symbols: " << circulant.minimum_reciprocal_condition() << " at ("
              << worst_frequency.x << ", " << worst_frequency.y << ", " << worst_frequency.z << ")" << std::endl;

    // 6. Calculate the analytical Mie field on the same mesh for comparison.
    auto analytical_solution = Research::VolDie::Mie::calculate_field_on_mesh(
        mesh, SPHERE_RADIUS + mesh.h() / 2., Types::complex_d{SPHERE_EPSILON, 0.0}, WAVE_NUMBER);
    mesh.setVectorData("analytical_solution", std::move(analytical_solution));

    // 7. GMRES sees B = P(I-KD)P C^{-1}P, so the circulant is applied from the right.
    using RightPreconditionedOperator =
        Math::LinAgl::Matrix::Wrappers::RightPreconditionedVolumeOperatorMatrixReplacement<FourierOperator, Circulant>;
    const RightPreconditionedOperator right_preconditioned_operator{fourier_operator, epsilon_minus_one, circulant};
    rhs = right_preconditioned_operator.modify_rhs_according_to_mask(rhs);

    // Research::solve returns y; recover the physical coefficients x = P C^{-1}P y afterwards.
    const Types::VectorXc transformed_solution = Research::solve<Eigen::GMRES>(
        right_preconditioned_operator, rhs, GMRES_MAX_ITERATIONS, GMRES_TOLERANCE, GMRES_RESTART);
    const Types::VectorXc solution = right_preconditioned_operator.recover_solution(transformed_solution);

    // Check the residual of the original masked equation rather than the transformed system.
    const Types::VectorXc contrast_solution = solution.cwiseProduct(right_preconditioned_operator.get_epsilon_vec());
    const Types::VectorXc original_operator_action =
        solution - (fourier_operator * contrast_solution).cwiseProduct(right_preconditioned_operator.get_mask());
    const Types::scalar relative_original_residual =
        (rhs - original_operator_action).norm() / std::max<Types::scalar>(rhs.norm(), 1e-30);
    std::cout << std::scientific << std::setprecision(6)
              << "Relative residual of the original masked system: " << relative_original_residual << std::defaultfloat
              << std::endl;

    // 8. Convert Galerkin coefficients into the electric field stored on mesh cells.
    std::vector<Types::Vector3c> field_on_mesh;
    field_on_mesh.reserve(mesh.getCells().size());
    for (std::size_t cell = 0; cell < mesh.getCells().size(); ++cell) {
        field_on_mesh.emplace_back(solution(3 * cell) * basis_function_module,
                                   solution(3 * cell + 1) * basis_function_module,
                                   solution(3 * cell + 2) * basis_function_module);
    }
    mesh.setVectorData("solution", std::move(field_on_mesh));

    // 9. Calculate and save numerical and analytical bistatic RSP cuts.
    constexpr int ANGLE_COUNT = 360;
    const auto get_tau_hh = [](Types::scalar phi) { return Types::Vector3d{std::sin(phi), 0.0, std::cos(phi)}; };
    const auto get_tau_vv = [](Types::scalar phi) { return Types::Vector3d{0.0, std::sin(phi), std::cos(phi)}; };

    const auto angle_view =
        std::views::iota(0, ANGLE_COUNT) | std::views::transform([](int index) { return index * M_PI / ANGLE_COUNT; });
    std::vector<Types::scalar> phis{angle_view.begin(), angle_view.end()};
    std::vector<Types::scalar> rsp_hh(ANGLE_COUNT);
    std::vector<Types::scalar> rsp_vv(ANGLE_COUNT);
    std::vector<Types::scalar> analytical_rsp_hh(ANGLE_COUNT);
    std::vector<Types::scalar> analytical_rsp_vv(ANGLE_COUNT);

#pragma omp parallel for
    for (auto &&[vv, hh, analytical_vv, analytical_hh, phi] :
         std::views::zip(rsp_vv, rsp_hh, analytical_rsp_vv, analytical_rsp_hh, phis)) {
        vv = 10.0 * std::log10(ESA::calculateRSP_kahan(get_tau_vv(phi), {WAVE_NUMBER.real(), 0.0}, "solution", mesh));
        hh = 10.0 * std::log10(ESA::calculateRSP_kahan(get_tau_hh(phi), {WAVE_NUMBER.real(), 0.0}, "solution", mesh));
        analytical_vv = 10.0 * std::log10(ESA::calculateRSP_kahan(get_tau_vv(phi), {WAVE_NUMBER.real(), 0.0},
                                                                  "analytical_solution", mesh));
        analytical_hh = 10.0 * std::log10(ESA::calculateRSP_kahan(get_tau_hh(phi), {WAVE_NUMBER.real(), 0.0},
                                                                  "analytical_solution", mesh));
    }

    const auto degree_view = angle_view | std::views::transform([](Types::scalar phi) { return phi * 180.0 / M_PI; });
    std::vector<Types::scalar> degrees{degree_view.begin(), degree_view.end()};
    std::ofstream rsp_vv_file{OUTPUT_DIRECTORY + "circulant_sphere_sigma_vv_" + std::to_string(NX) + ".csv"};
    std::ofstream rsp_hh_file{OUTPUT_DIRECTORY + "circulant_sphere_sigma_hh_" + std::to_string(NX) + ".csv"};
    std::ofstream analytical_rsp_vv_file{OUTPUT_DIRECTORY + "circulant_an_sphere_sigma_vv_" + std::to_string(NX) +
                                         ".csv"};
    std::ofstream analytical_rsp_hh_file{OUTPUT_DIRECTORY + "circulant_an_sphere_sigma_hh_" + std::to_string(NX) +
                                         ".csv"};
    Utils::to_csv(degrees, rsp_vv, "angle", "rsp", rsp_vv_file);
    Utils::to_csv(degrees, rsp_hh, "angle", "rsp", rsp_hh_file);
    Utils::to_csv(degrees, analytical_rsp_vv, "angle", "rsp", analytical_rsp_vv_file);
    Utils::to_csv(degrees, analytical_rsp_hh, "angle", "rsp", analytical_rsp_hh_file);

    // 10. Save the independent analytical far-field reference from the Mie series.
    auto mie_rsp =
        Research::VolDie::Mie::calculate_rsp(phis, SPHERE_RADIUS + mesh.h() / 2., Types::complex_d{SPHERE_EPSILON, 0.0}, WAVE_NUMBER);
    for (auto &value : mie_rsp.vv)
        value = 10.0 * std::log10(value);
    for (auto &value : mie_rsp.hh)
        value = 10.0 * std::log10(value);

    std::ofstream mie_rsp_vv_file{OUTPUT_DIRECTORY + "circulant_mie_sigma_vv.csv"};
    std::ofstream mie_rsp_hh_file{OUTPUT_DIRECTORY + "circulant_mie_sigma_hh.csv"};
    Utils::to_csv(degrees, mie_rsp.vv, "angle", "rsp", mie_rsp_vv_file);
    Utils::to_csv(degrees, mie_rsp.hh, "angle", "rsp", mie_rsp_hh_file);

    return 0;
}
