//
// Created by evgen on 18.01.2025.
//

#ifndef GENERALIZEDEQUATIONS_HPP
#define GENERALIZEDEQUATIONS_HPP

#include "mesh/SurfaceMesh.hpp"

#include "slae_generation/MatrixInplaceGeneration.hpp"

#include "types/Types.hpp"

#include "math/MathConstants.hpp"
#include "math/Productions.hpp"
#include "math/fields/SurfaceVectorField.hpp"
#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"
#include "math/integration/newton_cotess/Rectangular.hpp"

#include "operators/OperatorK.hpp"
#include "operators/OperatorR.hpp"

namespace WaveGuideWithActiveSection {

using namespace EMW;

/**
 * Вычисляем матрицу для "самодействия"
 */
inline EMW::Types::MatrixXc diagonal(const Mesh::SurfaceMesh &mesh_all, Types::scalar a, Types::complex_d k) {
    // вытаскиваем сетку для магнитных токов
    const auto mesh_zero = mesh_all.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    // описываем размеры итоговой матрицы
    const Types::index N = mesh_all.getCells().size();
    const Types::index K = mesh_zero.getCells().size();
    const Types::index size = 2 * (N + K);
    Types::MatrixXc A(size, size);

    const Types::complex_d epsilon{1., 0};
    const Types::complex_d mu{1., 0};

    // расчет коэффициента импеданса и вспомогательных констант
    const Types::complex_d beta = std::sqrt(k * k - EMW::Math::Constants::PI_square<Types::scalar>() / (a * a));
#ifdef OLD
    const Types::complex_d z = -k * Math::Constants::one_div_c / beta;
    const Types::complex_d with_e = Math::Constants::i * Math::Constants::mu_0_c / k;
    const Types::complex_d with_mu = Math::Constants::i * Math::Constants::e_0_c / k;
#else
    const Types::complex_d z = -k * mu / beta;
    const Types::complex_d with_e = Math::Constants::i / (k * epsilon);
    const Types::complex_d with_mu = Math::Constants::i / (k * mu);
#endif

    // генерируем необходимые подматрицы (это ссылки внутрь блоков)
    auto K_all_block = A.block(0, 0, 2 * N, 2 * N);
    auto R_0_block = A.block(0, 2 * N, 2 * N, 2 * K);
    auto R_all_block = A.block(2 * N, 0, 2 * K, 2 * N);
    auto K_0_block = A.block(2 * N, 2 * N, 2 * K, 2 * K);

    Matrix::getMatrixK_inplace(K_all_block, k, mesh_all);
    Matrix::getMatrixR_inplace(R_0_block, k, mesh_zero, mesh_all);
    Matrix::getMatrixK_inplace(K_0_block, k, mesh_zero);
    Matrix::getMatrixR_inplace(R_all_block, k, mesh_all, mesh_zero);

    // тривиальные операторы нужных размеров из уравнений
    // оператор векторного умножения на нормаль справа
    const Types::MatrixXc C = 0.5 * Matrix::getMatrixCrossNormal(k, mesh_zero);
    // единичный оператор
    const auto I_m = Matrix::getMatrixIdentity(k, mesh_zero);

    // 1) Дописываем слагаемые в уравнение на поверхности sigma_0
    // ВНИМАНИЕ: тут идёт существенный упор на то, что мы знаем, что sigma_0 лежит последняя в sigma_all
    R_0_block.block(N - K, 0, K, 2 * K) -= C.block(0, 0, K, 2 * K);
    R_0_block.block(2 * N - K, 0, K, 2 * K) -= C.block(K, 0, K, 2 * K);

    R_all_block.block(0, N - K, 2 * K, K) += C.block(0, 0, 2 * K, K);
    R_all_block.block(0, 2 * N - K, 2 * K, K) += C.block(0, K, 2 * K, K);

    // 2) Собираем матрицу
    K_all_block *= with_e;
    R_0_block *= -Types::complex_d{1, 0};
    R_all_block *= z;
    K_0_block *= z * with_mu;
    K_0_block += I_m;

    // 3) Готово
    return A;
}

// NOT AN INPLACE FUNCTION!!!!!!!!!!!!!!!!!!
/**
 * Вспомогательная функция для вычисления взаимодействия двух антенн
 * Сетки mesh_to_integrate и mesh_with_collocation_points не обязаны совпадать,
 * функция будет работать для произвольных сеток
 *
 * Результат этой функции при передаче индентичных сеток не эквивалентен расчету матрицы "самодействия",
 * так как в этом расчете не учитываются неаддитивные орепаторы, зависящие только от сетки
 * с точками коллокации (единичный оператор и оператор векторного умножения на нормаль справа)
 */
inline Types::MatrixXc submatrix(const Mesh::SurfaceMesh &mest_to_integrate,
                                 const Mesh::SurfaceMesh &mest_with_collocation_points, const Types::scalar a,
                                 const Types::complex_d k) {
    // описываем специфические части сеток
    // части с электрическими токами
    const Mesh::SurfaceMesh &mesh_to_integrate_all = mest_to_integrate;
    const Mesh::SurfaceMesh &mesh_to_integrate_zero =
        mest_to_integrate.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    const Mesh::SurfaceMesh &mesh_collocation_all = mest_with_collocation_points;
    const Mesh::SurfaceMesh &mesh_collocation_zero =
        mest_with_collocation_points.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);

    // Надо понять какие размеры будем иметь итоговая матрица
    const Types::index N_cols = mesh_to_integrate_all.getCells().size();
    const Types::index K_cols = mesh_to_integrate_zero.getCells().size();
    const Types::index cols = 2 * (N_cols + K_cols);
    const Types::index N_rows = mesh_collocation_all.getCells().size();
    const Types::index K_rows = mesh_collocation_zero.getCells().size();
    const Types::index rows = 2 * (N_rows + K_rows);
    Types::MatrixXc A(rows, cols);

    const Types::complex_d epsilon{1., 0};
    const Types::complex_d mu{1., 0};

    // расчет коэффициента импеданса и вспоногательных констант
    const Types::complex_d beta = std::sqrt(k * k - EMW::Math::Constants::PI_square<Types::scalar>() / (a * a));
#ifdef OLD
    const Types::complex_d z = -k * Math::Constants::one_div_c / beta;
    const Types::complex_d with_e = Math::Constants::i * Math::Constants::mu_0_c / k;
    const Types::complex_d with_mu = Math::Constants::i * Math::Constants::e_0_c / k;
#else
    const Types::complex_d z = -k * mu / beta;
    const Types::complex_d with_e = Math::Constants::i / (k * epsilon);
    const Types::complex_d with_mu = Math::Constants::i / (k * mu);
#endif

    // 1) генерируем необходимые подматрицы
    // матрица оператора K размерами 2 * N_rows x 2 * N_cols
    auto K_all = Matrix::getMatrixK(k, mesh_to_integrate_all, mesh_collocation_all);
    // матрица оператора R размерами 2 * N_rows x 2 * K_cols
    auto R_0 = Matrix::getMatrixR(k, mesh_to_integrate_zero, mesh_collocation_all);
    // матрица оператора K размерами 2 * K_rows x 2 * K_cols
    auto K_0 = Matrix::getMatrixK(k, mesh_to_integrate_zero, mesh_collocation_zero);
    // матрица оператора K размерами 2 * K_rows x 2 * N_cols
    auto R_all = Matrix::getMatrixR(k, mesh_to_integrate_all, mesh_collocation_zero);

    // 2) Неаддитивные по поверхности операторы отваливаются при расчете недиагональных подматриц
    // поэтому их тут не будет. Далее все идентично как и в случае одной антенны, блоки распихиваем точно так же

    // 3) Собираем матрицу
    A.block(0, 0, 2 * N_rows, 2 * N_cols) = with_e * K_all;
    A.block(0, 2 * N_cols, 2 * N_rows, 2 * K_cols) = -R_0;
    A.block(2 * N_rows, 0, 2 * K_rows, 2 * N_cols) = z * R_all;
    A.block(2 * N_rows, 2 * N_cols, 2 * K_rows, 2 * K_cols) = z * with_mu * K_0;

    // Готово
    return A;
}

inline Types::VectorXc row_of_submatrix(Types::index i,
                                        const Containers::vector<Mesh::IndexedCell> &mesh_to_integrate_all_cells,
                                        const Containers::vector<Mesh::IndexedCell> &mesh_to_integrate_zero_cells,
                                        const Containers::vector<Mesh::IndexedCell> &mesh_collocation_all_cells,
                                        const Containers::vector<Mesh::IndexedCell> &mesh_collocation_zero_cells,
                                        const Types::scalar a, const Types::complex_d k) {

    // Надо понять какие размеры будем иметь итоговая матрица
    const Types::index N_cols = mesh_to_integrate_all_cells.size();
    const Types::index K_cols = mesh_to_integrate_zero_cells.size();
    const Types::index cols = 2 * (N_cols + K_cols);
    const Types::index N_rows = mesh_collocation_all_cells.size();
    const Types::index K_rows = mesh_collocation_zero_cells.size();
    const Types::index rows = 2 * (N_rows + K_rows);
    // Надо понять какие константы будут фигурировать в расчете
    constexpr Types::complex_d epsilon{1., 0};
    constexpr Types::complex_d mu{1., 0};
    // расчет коэффициента импеданса и вспомогательных констант
    const Types::complex_d beta = std::sqrt(k * k - EMW::Math::Constants::PI_square<Types::scalar>() / (a * a));
#ifdef OLD
    const Types::complex_d z = -k * Math::Constants::one_div_c / beta;
    const Types::complex_d with_e = Math::Constants::i * Math::Constants::mu_0_c / k;
    const Types::complex_d with_mu = Math::Constants::i * Math::Constants::e_0_c / k;
#else
    const Types::complex_d z = -k * mu / beta;
    const Types::complex_d with_e = Math::Constants::i / (k * epsilon);
    const Types::complex_d with_mu = Math::Constants::i / (k * mu);
#endif
    // Расчет значения элемента непосредственно

    Types::VectorXc row = Types::VectorXc::Zero(cols);

    if (i < 2 * N_rows) {
        // это значит, что строка находится в верхней области
        // и имеет блочный вид ( with_e * К_all.row(i) | -R_0.row(i) )
        row.block(0, 0, 2 * N_cols, 1) =
            with_e * Matrix::getRowInMatrixK(i, k, mesh_to_integrate_all_cells, mesh_collocation_all_cells);
        row.block(2 * N_cols, 0, 2 * K_cols, 1) =
            -Matrix::getRowInMatrixR(i, k, mesh_to_integrate_zero_cells, mesh_collocation_all_cells);
    } else {
        // это значит, что строка находится в нижней половине
        const Types::index index_of_a_row = i - 2 * N_rows;
        // и имеет блочный вид ( z * R_all.row(i) | z * with_mu * K_0.row(i))
        row.block(0, 0, 2 * N_cols, 1) =
            z * Matrix::getRowInMatrixR(index_of_a_row, k, mesh_to_integrate_all_cells, mesh_collocation_zero_cells);
        row.block(2 * N_cols, 0, 2 * K_cols, 1) =
            z * with_mu * Matrix::getRowInMatrixK(index_of_a_row, k, mesh_to_integrate_zero_cells, mesh_collocation_zero_cells);
    }
    return row;
}

inline Types::VectorXc col_of_submatrix(Types::index i,
                                        const Containers::vector<Mesh::IndexedCell> &mesh_to_integrate_all_cells,
                                        const Containers::vector<Mesh::IndexedCell> &mesh_to_integrate_zero_cells,
                                        const Containers::vector<Mesh::IndexedCell> &mesh_collocation_all_cells,
                                        const Containers::vector<Mesh::IndexedCell> &mesh_collocation_zero_cells,
                                        const Types::scalar a, const Types::complex_d k) {

    // Надо понять какие размеры будем иметь итоговая матрица
    const Types::index N_cols = mesh_to_integrate_all_cells.size();
    const Types::index K_cols = mesh_to_integrate_zero_cells.size();
    const Types::index cols = 2 * (N_cols + K_cols);
    const Types::index N_rows = mesh_collocation_all_cells.size();
    const Types::index K_rows = mesh_collocation_zero_cells.size();
    const Types::index rows = 2 * (N_rows + K_rows);
    // Надо понять какие константы будут фигурировать в расчете
    constexpr Types::complex_d epsilon{1., 0};
    constexpr Types::complex_d mu{1., 0};
    // расчет коэффициента импеданса и вспомогательных констант
    const Types::complex_d beta = std::sqrt(k * k - EMW::Math::Constants::PI_square<Types::scalar>() / (a * a));
#ifdef OLD
    const Types::complex_d z = -k * Math::Constants::one_div_c / beta;
    const Types::complex_d with_e = Math::Constants::i * Math::Constants::mu_0_c / k;
    const Types::complex_d with_mu = Math::Constants::i * Math::Constants::e_0_c / k;
#else
    const Types::complex_d z = -k * mu / beta;
    const Types::complex_d with_e = Math::Constants::i / (k * epsilon);
    const Types::complex_d with_mu = Math::Constants::i / (k * mu);
#endif
    // Расчет значения элемента непосредственно

    Types::VectorXc col = Types::VectorXc::Zero(rows);

    if (i < 2 * N_cols) {
        // это значит, что колонка находится в левой области
        // и имеет блочный вид ( with_e * К_all.col(i) | z * R_0.col(i) )
        col.block(0, 0, 2 * N_rows, 1) =
            with_e * Matrix::getColumnInMatrixK(i, k, mesh_to_integrate_all_cells, mesh_collocation_all_cells);
        col.block(2 * N_rows, 0, 2 * K_rows, 1) =
            z * Matrix::getColumnInMatrixR(i, k, mesh_to_integrate_all_cells, mesh_collocation_zero_cells);
    } else {
        // это значит, что строка находится в правой области
        const Types::index index_of_a_col = i - 2 * N_cols;
        // и имеет блочный вид ( -R_0.col(i) | z * with_mu * K_0.col(i))
        col.block(0, 0, 2 * N_rows, 1) =
            -Matrix::getColumnInMatrixR(index_of_a_col, k, mesh_to_integrate_zero_cells, mesh_collocation_all_cells);
        col.block(2 * N_rows, 0, 2 * K_rows, 1) =
            z * with_mu * Matrix::getColumnInMatrixK(index_of_a_col, k, mesh_to_integrate_zero_cells, mesh_collocation_zero_cells);
    }
    return col;
}


inline Types::VectorXc getRhsBase(const Mesh::SurfaceMesh &mesh_all, Types::scalar a, Types::complex_d k) {

    // Собираем сабсетку с активным сечением
    const auto &mesh_zero = mesh_all.getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION);
    // Собираем функцию поля на активном сечении волновода
    const Types::complex_d beta = std::sqrt(k * k - (EMW::Math::Constants::PI_square<Types::scalar>() / (a * a)));
    const auto get_e_h10_mode = [&k, &a, &beta](const Mesh::point_t &x) {
        const auto pi = Math::Constants::PI<Types::scalar>();
        const Types::complex_d mult = Math::Constants::i * (a / pi) * k / Math::Constants::e_0_c;
        const Types::complex_d exp = std::exp(Math::Constants::i * beta * (x.z() + 0.17));
        const Types::scalar cos = std::cos(pi * x.x() / a);
        return Types::Vector3c{Types::complex_d{0, 0}, mult * cos * exp, Types::complex_d{0, 0}};
    };

    const Types::index N = mesh_all.getCells().size();
    const Types::index K = mesh_zero.getCells().size();

    const Math::SurfaceVectorField direct_e_field{mesh_zero, get_e_h10_mode};
    const Types::VectorXc non_zer0_rhs = -2 * direct_e_field.normalCrossField().asVector();

    Types::VectorXc res = Types::VectorXc::Zero(2 * N + 2 * K);
    res.block(2 * N, 0, 2 * K, 1) = non_zer0_rhs;
    return res;
}

template <Types::index N>
Types::VectorXc getRhs(std::array<Types::complex_d, N> phases, const Mesh::SurfaceMesh &mesh_all, Types::scalar a,
                       Types::complex_d k) {
    const auto base_rhs = getRhsBase(mesh_all, a, k);

    Types::VectorXc res{N * base_rhs.size()};

    for (int i = 0; i < N; ++i) {
        res.block(i * base_rhs.rows(), 0, base_rhs.rows(), 1) = phases[i] * base_rhs;
    }
    return res;
}

using diagonal_t = decltype(diagonal);
using submatrix_t = decltype(submatrix);

// ---- Probably deprecated ---- //
inline Types::complex_d element_of_submatrix(Types::index i, Types::index j,
                                             const Containers::vector<Mesh::IndexedCell> &mesh_to_integrate_all_cells,
                                             const Containers::vector<Mesh::IndexedCell> &mesh_to_integrate_zero_cells,
                                             const Containers::vector<Mesh::IndexedCell> &mesh_collocation_all_cells,
                                             const Containers::vector<Mesh::IndexedCell> &mesh_collocation_zero_cells,
                                             const Types::scalar a, const Types::complex_d k) {
    // Надо понять какие размеры будем иметь итоговая матрица
    const Types::index N_cols = mesh_to_integrate_all_cells.size();
    const Types::index K_cols = mesh_to_integrate_zero_cells.size();
    const Types::index cols = 2 * (N_cols + K_cols);
    const Types::index N_rows = mesh_collocation_all_cells.size();
    const Types::index K_rows = mesh_collocation_zero_cells.size();
    const Types::index rows = 2 * (N_rows + K_rows);
    // Надо понять какие константы будут фигурировать в расчете
    constexpr Types::complex_d epsilon{1., 0};
    constexpr Types::complex_d mu{1., 0};
    // расчет коэффициента импеданса и вспомогательных констант
    const Types::complex_d beta = std::sqrt(k * k - EMW::Math::Constants::PI_square<Types::scalar>() / (a * a));
#ifdef OLD
    const Types::complex_d z = -k * Math::Constants::one_div_c / beta;
    const Types::complex_d with_e = Math::Constants::i * Math::Constants::mu_0_c / k;
    const Types::complex_d with_mu = Math::Constants::i * Math::Constants::e_0_c / k;
#else
    const Types::complex_d z = -k * mu / beta;
    const Types::complex_d with_e = Math::Constants::i / (k * epsilon);
    const Types::complex_d with_mu = Math::Constants::i / (k * mu);
#endif
    // Расчет значения элемента непосредственно

    if (i < 2 * N_rows) {
        const auto cell_i = mesh_collocation_all_cells[i % N_rows];
        const Types::index i_for_taus = i / N_rows;
        if (j < 2 * N_cols) {
            const auto cell_j = mesh_to_integrate_all_cells[j % N_cols];
            // код из функции со сбором матрицы (a bit modified)
            const Types::index j_for_taus = j / N_cols;
            const Types::Matrix3c int0 = Matrix::DiscreteK::getZeroPartIntegral(cell_i, cell_j, k);
            const Types::complex_d int1_k2 = k * k * Matrix::DiscreteK::getFirstPartIntegral(cell_i, cell_j, k);
            const Types::complex_d a_0 = cell_i.tau[i_for_taus].transpose() * int0 * cell_j.tau[j_for_taus];
            const Types::complex_d a_1 = Math::quasiDot(cell_i.tau[i_for_taus], cell_j.tau[j_for_taus]) * int1_k2;
            return with_e * (a_0 + a_1);

        } else {
            Types::index local_j = j - 2 * N_cols;
            const Types::index j_for_taus = local_j / K_cols;
            const auto cell_j = mesh_to_integrate_zero_cells[local_j % K_cols];
            const Types::Vector3c integral =
                OperatorR::detail::forMatrix::commonIntegralPart<DefiniteIntegrals::GaussLegendre::Quadrature<2, 2>>(
                    cell_j, cell_i.collPoint_, k);
            return -Math::quasiDot(integral, cell_j.tau[j_for_taus].cross(cell_i.tau[i_for_taus]));
        }
    } else {
        Types::index local_i = i - 2 * N_rows;
        const auto cell_i = mesh_collocation_zero_cells[local_i % K_rows];
        const Types::index i_for_taus = local_i / K_rows;
        if (j < 2 * N_cols) {
            const auto cell_j = mesh_to_integrate_all_cells[j % N_cols];
            // код из функции со сбором матрицы (a bit modified)
            const Types::index j_for_taus = j / N_cols;
            const Types::Vector3c integral =
                OperatorR::detail::forMatrix::commonIntegralPart<DefiniteIntegrals::GaussLegendre::Quadrature<2, 2>>(
                    cell_j, cell_i.collPoint_, k);
            return z * Math::quasiDot(integral, cell_j.tau[j_for_taus].cross(cell_i.tau[i_for_taus]));

        } else {
            Types::index local_j = j - 2 * N_cols;
            const Types::index j_for_taus = local_j / K_cols;
            const auto cell_j = mesh_to_integrate_zero_cells[local_j % K_cols];
            const Types::Matrix3c int0 = Matrix::DiscreteK::getZeroPartIntegral(cell_i, cell_j, k);
            const Types::complex_d int1_k2 = k * k * Matrix::DiscreteK::getFirstPartIntegral(cell_i, cell_j, k);
            const Types::complex_d a_0 = cell_i.tau[i_for_taus].transpose() * int0 * cell_j.tau[j_for_taus];
            const Types::complex_d a_1 = Math::quasiDot(cell_i.tau[i_for_taus], cell_j.tau[j_for_taus]) * int1_k2;
            return z * with_mu * (a_0 + a_1);
        }
    }
}
}

#endif //GENERALIZEDEQUATIONS_HPP
