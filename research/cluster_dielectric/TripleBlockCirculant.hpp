#ifndef RESEARCH_TRIPLE_BLOCK_CIRCULANT_HPP
#define RESEARCH_TRIPLE_BLOCK_CIRCULANT_HPP

#include "math/fourier/TripleToeplitz3x3Fourier.hpp"
#include "types/Types.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Research::VolDie {

struct CirculantIndex3D {
    EMW::Types::index x;
    EMW::Types::index y;
    EMW::Types::index z;
};

namespace Detail {

inline EMW::Types::index flatten_circulant_index(EMW::Types::index x, EMW::Types::index y,
                                                  EMW::Types::index z, EMW::Types::index nx,
                                                  EMW::Types::index ny) noexcept {
    return x + nx * (y + ny * z);
}

inline CirculantIndex3D unflatten_circulant_index(EMW::Types::index index, EMW::Types::index nx,
                                                   EMW::Types::index ny) noexcept {
    const EMW::Types::index x = index % nx;
    index /= nx;
    const EMW::Types::index y = index % ny;
    return {x, y, index / ny};
}

inline std::pair<EMW::Types::index, EMW::Types::index>
row_col_for_circulant_shift(EMW::Types::integer shift) {
    if (shift >= 0)
        return {static_cast<EMW::Types::index>(shift), 0};
    return {0, static_cast<EMW::Types::index>(-shift)};
}

// OperatorK has two three-level Toeplitz storage implementations in this
// project. This accessor supports both the new six-index API and the legacy
// nested get_block(...).get_block(...).get_block(...) API.
template <typename TripleToeplitzMatrix>
decltype(auto) get_toeplitz_3x3_block(const TripleToeplitzMatrix &matrix, EMW::Types::index row_z,
                                      EMW::Types::index col_z, EMW::Types::index row_y,
                                      EMW::Types::index col_y, EMW::Types::index row_x,
                                      EMW::Types::index col_x) {
    if constexpr (requires { matrix.get_block(row_z, col_z, row_y, col_y, row_x, col_x); }) {
        return matrix.get_block(row_z, col_z, row_y, col_y, row_x, col_x);
    } else {
        return matrix.get_block(row_z, col_z).get_block(row_y, col_y).get_block(row_x, col_x);
    }
}

template <typename Scalar, typename TripleToeplitzMatrix>
Eigen::Matrix<Scalar, 3, 3> system_block_for_shift(const TripleToeplitzMatrix &operator_k,
                                                    EMW::Types::integer dx, EMW::Types::integer dy,
                                                    EMW::Types::integer dz, Scalar contrast) {
    const auto [row_x, col_x] = row_col_for_circulant_shift(dx);
    const auto [row_y, col_y] = row_col_for_circulant_shift(dy);
    const auto [row_z, col_z] = row_col_for_circulant_shift(dz);

    Eigen::Matrix<Scalar, 3, 3> block =
        -contrast * get_toeplitz_3x3_block(operator_k, row_z, col_z, row_y, col_y, row_x, col_x);
    if (dx == 0 && dy == 0 && dz == 0)
        block.diagonal().array() += Scalar{1};
    return block;
}

} // namespace Detail

/**
 * Builds the first block column of the Frobenius-optimal three-level Chan
 * circulant for A = I - contrast * K.
 *
 * For a circular offset r, the Toeplitz shifts congruent to r modulo the grid
 * size are averaged with weights equal to the relative number of entries that
 * carry each shift. In three dimensions these weights factor by axis.
 */
template <typename Scalar, typename TripleToeplitzMatrix>
std::vector<Eigen::Matrix<Scalar, 3, 3>>
build_chan_circulant_first_block_column(const TripleToeplitzMatrix &operator_k, EMW::Types::index nx,
                                         EMW::Types::index ny, EMW::Types::index nz, Scalar contrast) {
    using Block = Eigen::Matrix<Scalar, 3, 3>;
    std::vector<Block> first_column(nx * ny * nz, Block::Zero());

#pragma omp parallel for collapse(2) schedule(static)
    for (EMW::Types::index rz = 0; rz < nz; ++rz) {
        for (EMW::Types::index ry = 0; ry < ny; ++ry) {
            for (EMW::Types::index rx = 0; rx < nx; ++rx) {
                Block circulant_block = Block::Zero();

                const EMW::Types::index aliases_z = rz == 0 ? 1 : 2;
                const EMW::Types::index aliases_y = ry == 0 ? 1 : 2;
                const EMW::Types::index aliases_x = rx == 0 ? 1 : 2;

                for (EMW::Types::index alias_z = 0; alias_z < aliases_z; ++alias_z) {
                    const EMW::Types::integer dz =
                        alias_z == 0 ? static_cast<EMW::Types::integer>(rz)
                                     : static_cast<EMW::Types::integer>(rz) - static_cast<EMW::Types::integer>(nz);
                    const auto weight_z = 1.0 - static_cast<double>(std::abs(dz)) / static_cast<double>(nz);

                    for (EMW::Types::index alias_y = 0; alias_y < aliases_y; ++alias_y) {
                        const EMW::Types::integer dy = alias_y == 0
                                                           ? static_cast<EMW::Types::integer>(ry)
                                                           : static_cast<EMW::Types::integer>(ry) -
                                                                 static_cast<EMW::Types::integer>(ny);
                        const auto weight_y = 1.0 - static_cast<double>(std::abs(dy)) / static_cast<double>(ny);

                        for (EMW::Types::index alias_x = 0; alias_x < aliases_x; ++alias_x) {
                            const EMW::Types::integer dx = alias_x == 0
                                                               ? static_cast<EMW::Types::integer>(rx)
                                                               : static_cast<EMW::Types::integer>(rx) -
                                                                     static_cast<EMW::Types::integer>(nx);
                            const auto weight_x = 1.0 - static_cast<double>(std::abs(dx)) / static_cast<double>(nx);

                            circulant_block += static_cast<typename Eigen::NumTraits<Scalar>::Real>(
                                                   weight_x * weight_y * weight_z) *
                                               Detail::system_block_for_shift<Scalar>(operator_k, dx, dy, dz,
                                                                                       contrast);
                        }
                    }
                }

                first_column[Detail::flatten_circulant_index(rx, ry, rz, nx, ny)] = circulant_block;
            }
        }
    }
    return first_column;
}

/**
 * Three-level block-circulant matrix with dense 3x3 inner blocks.
 *
 * Both C*x and C^{-1}*x are evaluated through an unpadded three-dimensional
 * FFT. At every Fourier node the operation reduces to multiplication by (or
 * the inverse of) one dense 3x3 matrix symbol.
 */
template <typename Scalar> class TripleBlockCirculant3x3FFT {
  public:
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Block = Eigen::Matrix<Scalar, 3, 3>;
    using LocalVector = Eigen::Matrix<Scalar, 3, 1>;
    using Vector = EMW::Types::VectorX<Scalar>;

    static constexpr EMW::Types::index BLOCK_SIZE = 3;

  private:
    static_assert(Eigen::NumTraits<Scalar>::IsComplex, "The FFT circulant scalar must be complex");
    static_assert(std::is_same_v<Scalar, std::complex<RealScalar>>,
                  "The scalar must be std::complex<RealScalar>");

    static constexpr EMW::Types::index SYMBOL_ENTRIES = BLOCK_SIZE * BLOCK_SIZE;

    EMW::Types::index nx_;
    EMW::Types::index ny_;
    EMW::Types::index nz_;
    std::vector<Block> first_block_column_;
    std::vector<Block> symbol_;
    std::vector<Block> inverse_symbol_;
    RealScalar minimum_reciprocal_condition_ = std::numeric_limits<RealScalar>::infinity();
    EMW::Types::index worst_frequency_index_ = 0;

    [[nodiscard]] EMW::Types::index grid_size() const noexcept { return nx_ * ny_ * nz_; }

    template <std::size_t ComponentCount>
    void fft3_inplace(std::vector<std::array<Scalar, ComponentCount>> &data, bool inverse) const {
        if (data.size() != grid_size())
            throw std::invalid_argument("Incorrect data size in TripleBlockCirculant3x3FFT::fft3_inplace");

        const EMW::Types::index max_axis_size = std::max({nx_, ny_, nz_});

#pragma omp parallel
        {
            Eigen::FFT<RealScalar> fft;
            std::vector<Scalar> input_line(max_axis_size);
            std::vector<Scalar> output_line(max_axis_size);

#pragma omp for collapse(2) schedule(static)
            for (EMW::Types::index z = 0; z < nz_; ++z) {
                for (EMW::Types::index y = 0; y < ny_; ++y) {
                    for (EMW::Types::index component = 0; component < ComponentCount; ++component) {
                        for (EMW::Types::index x = 0; x < nx_; ++x)
                            input_line[x] = data[Detail::flatten_circulant_index(x, y, z, nx_, ny_)][component];
                        if (inverse)
                            fft.inv(output_line.data(), input_line.data(), nx_);
                        else
                            fft.fwd(output_line.data(), input_line.data(), nx_);
                        for (EMW::Types::index x = 0; x < nx_; ++x)
                            data[Detail::flatten_circulant_index(x, y, z, nx_, ny_)][component] = output_line[x];
                    }
                }
            }

#pragma omp for collapse(2) schedule(static)
            for (EMW::Types::index z = 0; z < nz_; ++z) {
                for (EMW::Types::index x = 0; x < nx_; ++x) {
                    for (EMW::Types::index component = 0; component < ComponentCount; ++component) {
                        for (EMW::Types::index y = 0; y < ny_; ++y)
                            input_line[y] = data[Detail::flatten_circulant_index(x, y, z, nx_, ny_)][component];
                        if (inverse)
                            fft.inv(output_line.data(), input_line.data(), ny_);
                        else
                            fft.fwd(output_line.data(), input_line.data(), ny_);
                        for (EMW::Types::index y = 0; y < ny_; ++y)
                            data[Detail::flatten_circulant_index(x, y, z, nx_, ny_)][component] = output_line[y];
                    }
                }
            }

#pragma omp for collapse(2) schedule(static)
            for (EMW::Types::index y = 0; y < ny_; ++y) {
                for (EMW::Types::index x = 0; x < nx_; ++x) {
                    for (EMW::Types::index component = 0; component < ComponentCount; ++component) {
                        for (EMW::Types::index z = 0; z < nz_; ++z)
                            input_line[z] = data[Detail::flatten_circulant_index(x, y, z, nx_, ny_)][component];
                        if (inverse)
                            fft.inv(output_line.data(), input_line.data(), nz_);
                        else
                            fft.fwd(output_line.data(), input_line.data(), nz_);
                        for (EMW::Types::index z = 0; z < nz_; ++z)
                            data[Detail::flatten_circulant_index(x, y, z, nx_, ny_)][component] = output_line[z];
                    }
                }
            }
        }
    }

    void build_symbols() {
        std::vector<std::array<Scalar, SYMBOL_ENTRIES>> transformed_column(grid_size());
        for (EMW::Types::index grid_index = 0; grid_index < grid_size(); ++grid_index) {
            for (EMW::Types::index row = 0; row < BLOCK_SIZE; ++row) {
                for (EMW::Types::index col = 0; col < BLOCK_SIZE; ++col) {
                    transformed_column[grid_index][row * BLOCK_SIZE + col] =
                        first_block_column_[grid_index](row, col);
                }
            }
        }
        fft3_inplace(transformed_column, false);

        symbol_.resize(grid_size());
        inverse_symbol_.resize(grid_size());
        for (EMW::Types::index frequency_index = 0; frequency_index < grid_size(); ++frequency_index) {
            for (EMW::Types::index row = 0; row < BLOCK_SIZE; ++row) {
                for (EMW::Types::index col = 0; col < BLOCK_SIZE; ++col) {
                    symbol_[frequency_index](row, col) =
                        transformed_column[frequency_index][row * BLOCK_SIZE + col];
                }
            }

            Eigen::FullPivLU<Block> factorization{symbol_[frequency_index]};
            const RealScalar reciprocal_condition = factorization.rcond();
            if (!factorization.isInvertible() || !std::isfinite(reciprocal_condition) ||
                reciprocal_condition <= RealScalar{0}) {
                const auto frequency = Detail::unflatten_circulant_index(frequency_index, nx_, ny_);
                throw std::runtime_error("Singular circulant symbol at frequency (" +
                                         std::to_string(frequency.x) + ", " + std::to_string(frequency.y) + ", " +
                                         std::to_string(frequency.z) + ")");
            }

            if (reciprocal_condition < minimum_reciprocal_condition_) {
                minimum_reciprocal_condition_ = reciprocal_condition;
                worst_frequency_index_ = frequency_index;
            }
            inverse_symbol_[frequency_index] = factorization.solve(Block::Identity());
        }
    }

    [[nodiscard]] Vector apply_symbol(const Vector &input, const std::vector<Block> &frequency_blocks) const {
        if (static_cast<EMW::Types::index>(input.size()) != rows())
            throw std::invalid_argument("Incorrect vector size for block-circulant multiplication");

        std::vector<std::array<Scalar, BLOCK_SIZE>> transformed_vector(grid_size());
        for (EMW::Types::index cell = 0; cell < grid_size(); ++cell) {
            for (EMW::Types::index component = 0; component < BLOCK_SIZE; ++component)
                transformed_vector[cell][component] = input(BLOCK_SIZE * cell + component);
        }

        fft3_inplace(transformed_vector, false);

#pragma omp parallel for schedule(static)
        for (EMW::Types::index frequency_index = 0; frequency_index < grid_size(); ++frequency_index) {
            LocalVector frequency_value;
            for (EMW::Types::index component = 0; component < BLOCK_SIZE; ++component)
                frequency_value(component) = transformed_vector[frequency_index][component];

            const LocalVector transformed_value = frequency_blocks[frequency_index] * frequency_value;
            for (EMW::Types::index component = 0; component < BLOCK_SIZE; ++component)
                transformed_vector[frequency_index][component] = transformed_value(component);
        }

        fft3_inplace(transformed_vector, true);

        Vector result(rows());
        for (EMW::Types::index cell = 0; cell < grid_size(); ++cell) {
            for (EMW::Types::index component = 0; component < BLOCK_SIZE; ++component)
                result(BLOCK_SIZE * cell + component) = transformed_vector[cell][component];
        }
        return result;
    }

  public:
    TripleBlockCirculant3x3FFT(EMW::Types::index nx, EMW::Types::index ny, EMW::Types::index nz,
                               std::vector<Block> first_block_column)
        : nx_(nx), ny_(ny), nz_(nz), first_block_column_(std::move(first_block_column)) {
        // Eigen's default FFT backend does not support transforms of length one.
        if (nx_ < 2 || ny_ < 2 || nz_ < 2 || first_block_column_.size() != grid_size())
            throw std::invalid_argument("Each circulant grid dimension must contain at least two cells");
        build_symbols();
    }

    [[nodiscard]] EMW::Types::index rows() const noexcept { return BLOCK_SIZE * grid_size(); }
    [[nodiscard]] EMW::Types::index cols() const noexcept { return rows(); }

    [[nodiscard]] Vector multiply(const Vector &input) const { return apply_symbol(input, symbol_); }
    [[nodiscard]] Vector solve(const Vector &input) const { return apply_symbol(input, inverse_symbol_); }
    [[nodiscard]] Vector operator*(const Vector &input) const { return multiply(input); }

    // Slow reference multiplication used to verify the FFT convention on small grids.
    [[nodiscard]] Vector multiply_direct(const Vector &input) const {
        if (static_cast<EMW::Types::index>(input.size()) != rows())
            throw std::invalid_argument("Incorrect vector size for direct block-circulant multiplication");

        Vector result = Vector::Zero(rows());
#pragma omp parallel for schedule(static)
        for (EMW::Types::index output_cell = 0; output_cell < grid_size(); ++output_cell) {
            const auto output = Detail::unflatten_circulant_index(output_cell, nx_, ny_);
            LocalVector value = LocalVector::Zero();

            for (EMW::Types::index input_cell = 0; input_cell < grid_size(); ++input_cell) {
                const auto input_index = Detail::unflatten_circulant_index(input_cell, nx_, ny_);
                const EMW::Types::index rx = (output.x + nx_ - input_index.x) % nx_;
                const EMW::Types::index ry = (output.y + ny_ - input_index.y) % ny_;
                const EMW::Types::index rz = (output.z + nz_ - input_index.z) % nz_;
                value += first_block_column_[Detail::flatten_circulant_index(rx, ry, rz, nx_, ny_)] *
                         input.template segment<3>(BLOCK_SIZE * input_cell);
            }
            result.template segment<3>(BLOCK_SIZE * output_cell) = value;
        }
        return result;
    }

    [[nodiscard]] RealScalar minimum_reciprocal_condition() const noexcept {
        return minimum_reciprocal_condition_;
    }

    [[nodiscard]] CirculantIndex3D worst_conditioned_frequency() const noexcept {
        return Detail::unflatten_circulant_index(worst_frequency_index_, nx_, ny_);
    }
};

} // namespace Research::VolDie

#endif // RESEARCH_TRIPLE_BLOCK_CIRCULANT_HPP
