//
// Strong scalability test for the 3D Fourier triple-Toeplitz matvec.
//

#include "math/fourier/TripleToeplitz3x3Fourier.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>

#include <omp.h>

using namespace EMW;

namespace {

using fourier_t = Math::Fourier::TripleToeplitz3x3FourierParallel<Types::complex_d>;
using tensor_t = fourier_t::tensor_type;

constexpr Types::index CELLS_PER_AXIS = 128;
constexpr int MAX_THREADS = 16;

class deterministic_random {
    std::uint64_t state_ = 0x8a5cd789635d2dffULL;

  public:
    [[nodiscard]] Types::scalar symmetric(Types::scalar scale) noexcept {
        // A small, fast and reproducible xorshift generator is sufficient for synthetic test data.
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        const std::uint64_t value = state_ * 0x2545f4914f6cdd1dULL;
        const Types::scalar unit = static_cast<Types::scalar>(value >> 11) * 0x1.0p-53;
        return scale * (2.0 * unit - 1.0);
    }
};

[[nodiscard]] tensor_t make_test_matrix(Types::index cells_per_axis, deterministic_random &random) {
    const Types::index levels_per_axis = 2 * cells_per_axis - 1;
    const Types::scalar cells_count = static_cast<Types::scalar>(cells_per_axis) *
                                      static_cast<Types::scalar>(cells_per_axis) *
                                      static_cast<Types::scalar>(cells_per_axis);

    // Every output component contains 3 * cells_count summands. This scaling keeps |A*x|
    // bounded by a small constant for input components with real and imaginary parts in [-1, 1].
    const Types::scalar coefficient_scale = 0.1 / (3.0 * cells_count);
    tensor_t matrix(levels_per_axis, levels_per_axis, levels_per_axis);

    for (Types::index lz = 0; lz < levels_per_axis; ++lz) {
        for (Types::index ly = 0; ly < levels_per_axis; ++ly) {
            for (Types::index lx = 0; lx < levels_per_axis; ++lx) {
                for (Types::index row = 0; row < 3; ++row) {
                    for (Types::index col = 0; col < 3; ++col) {
                        matrix(lx, ly, lz, row, col) = {
                            random.symmetric(coefficient_scale), random.symmetric(coefficient_scale)};
                    }
                }
            }
        }
    }

    return matrix;
}

[[nodiscard]] Types::VectorXc make_test_vector(Types::index size, deterministic_random &random) {
    Types::VectorXc vector(size);
    for (Types::index i = 0; i < size; ++i) {
        vector(i) = {random.symmetric(1.0), random.symmetric(1.0)};
    }
    return vector;
}

} // namespace

int main() {
    Eigen::setNbThreads(1);

    const int available_threads = omp_get_max_threads();
    const int last_thread_count = std::min(MAX_THREADS, available_threads);

    deterministic_random random;
    auto levels = make_test_matrix(CELLS_PER_AXIS, random);

    // Kernel embedding and its 3D Fourier transform are preparation steps and are not timed.
    const fourier_t matrix{std::move(levels)};
    const Types::VectorXc vector = make_test_vector(matrix.cols(), random);

    std::cout << "Cells per axis: " << CELLS_PER_AXIS << '\n';
    std::cout << "Unknowns: " << matrix.cols() << '\n';
    std::cout << "Max threads available: " << available_threads << std::endl;

    for (int n = 1; n <= last_thread_count; ++n) {
        omp_set_num_threads(n);

        // Warm up the OpenMP team and thread-local FFT working buffers for this thread count.
        {
            const auto warmup_result = matrix * vector;
            if (!warmup_result.allFinite()) {
                std::cerr << "Non-finite warm-up result for " << n << " threads" << std::endl;
                return EXIT_FAILURE;
            }
        }

        const auto start = std::chrono::steady_clock::now();
        const auto result = matrix * vector;
        const auto end = std::chrono::steady_clock::now();

        if (!result.allFinite()) {
            std::cerr << "Non-finite result for " << n << " threads" << std::endl;
            return EXIT_FAILURE;
        }

        const auto elapsed = std::chrono::duration<Types::scalar, std::milli>(end - start);
        std::cout << "# " << n << ": " << elapsed.count() << " milliseconds"
                  << ", ||A*x|| = " << result.norm() << std::endl;
    }
}
